# Native Memory Leak Debugging Process

## Hybrid Java (JVM) + JNI + C/C++ + CPython Systems

This document describes a structured debugging methodology for identifying and resolving native memory leaks in systems that integrate:

- Java (JVM)
- Java NIO / DirectByteBuffer (off-heap)
- JNI bridge
- Native C/C++ libraries (e.g., biometric SDKs)
- CPython embedded inside the JVM process

Core principle:

> The JVM GC does NOT manage native malloc memory.  
> CPython does NOT manage JVM memory.  
> Native C libraries do NOT integrate with either GC automatically.

Each runtime has its own memory model.

---

# 1️⃣ Memory Domains in Hybrid Systems

In a Java + C + CPython environment, memory is divided into:

1. JVM Heap (managed by JVM GC)
2. JVM Native Memory (DirectByteBuffer, JNI internal allocations)
3. CPython Heap (PyObject allocations via PyMem / obmalloc)
4. External Native Heap (malloc/free from C libraries)

Only #1 is visible via Java heap dumps.

Leaks often occur in #2, #3, or #4.

---

# 2️⃣ Initial Symptom Pattern

Common leak symptoms:

- JVM heap remains stable
- No OutOfMemoryError (heap)
- RSS steadily increasing
- Process eventually killed by OS
- No visible leak in Java profiling tools

This strongly indicates:

> Native memory leak (outside JVM heap)

---

# 3️⃣ First-Level Inspection: JVM Native Tracking

Enable JVM native tracking:

    -XX:NativeMemoryTracking=detail
    -XX:+UnlockDiagnosticVMOptions
    -XX:+PrintNMTStatistics

At runtime:

    jcmd <pid> VM.native_memory summary

If:

- JVM native stable
- RSS still growing

Then leak is likely in:

- CPython allocations
- External C library malloc
- JNI misuse

---

# 4️⃣ CPython Memory Model

CPython uses:

- Reference counting (primary)
- Cyclic GC (secondary)
- obmalloc small-object allocator
- Underlying system malloc for larger blocks

Important facts:

- CPython memory is NOT visible to JVM GC.
- If Py_DECREF is not called properly, memory leaks.
- If reference cycles exist and GC is disabled, leaks may occur.
- If native C extensions allocate memory without freeing, leaks occur.

---

# 5️⃣ Common CPython Leak Patterns in JVM Embedding

## Pattern A — Missing Py_DECREF

Example:

- Function returns new reference
- Java wrapper does not DECREF
- Object permanently retained

Result:
- CPython heap growth
- RSS growth

---

## Pattern B — Borrowed vs New Reference Confusion

If you DECREF a borrowed reference:

- Use-after-free
- Segfault

If you fail to DECREF a new reference:

- Leak

---

## Pattern C — GIL Misuse

Calling CPython API without holding the GIL:

- Memory corruption
- Random crashes
- Inconsistent refcount state

Correct pattern:

    PyGILState_Ensure()
    ... Python calls ...
    PyGILState_Release()

---

## Pattern D — CPython C Extension Leak

If a C extension used by Python:

- Allocates via malloc
- Never frees

Then CPython GC cannot help.

Must instrument malloc.

---

# 6️⃣ Instrumenting Native Allocations

To debug C-level and CPython-level leaks, replace or instrument malloc.

Two production-grade allocators:

- https://google.github.io/tcmalloc/
- https://jemalloc.net/

---

# 7️⃣ Using tcmalloc (Java + CPython)

Example:

    LD_PRELOAD=/path/to/libtcmalloc.so java -jar app.jar

Enable profiling:

    HEAPPROFILE=./heap_profile java -jar app.jar

This captures allocations from:

- JNI
- CPython
- Native C libraries

Allows identification of:

- Largest allocation site
- Unreleased allocations
- Growth patterns

---

# 8️⃣ Using jemalloc

Example:

    LD_PRELOAD=/path/to/libjemalloc.so MALLOC_CONF=prof:true java -jar app.jar

Generates allocation profiles including:

- CPython obmalloc fallbacks
- C extension allocations
- JNI native allocations

Useful for:

- Long-running production leak analysis
- Allocation flamegraphs

---

# 9️⃣ Debugging CPython-Specific Leaks

## Enable Python Fault Handler

Inside embedded interpreter:

    import faulthandler
    faulthandler.enable()

This ensures Python-level traceback on crash.

---

## Debug Build of CPython

Compile Python with:

    ./configure --with-pydebug
    make

Benefits:

- Refcount debug
- Extra assertions
- Memory debug hooks

---

## Track Reference Counts

In debug builds:

    sys.gettotalrefcount()

Useful to detect refcount growth per request cycle.

---

# 🔟 JNI ↔ CPython ↔ C Ownership Rules

Every allocation must have:

- Single clear owner
- Single clear destroy function
- Explicit lifecycle contract

Example contract:

- C allocates → Java must call destroy()
- CPython returns new reference → Java must DECREF
- JNI creates GlobalRef → must DeleteGlobalRef

Never:

- Assume GC will clean native memory
- Rely on finalizers
- Share implicit ownership

---

# 1️⃣1️⃣ Structured Debugging Workflow

Step-by-step:

1. Observe RSS growth.
2. Confirm JVM heap stable.
3. Enable NativeMemoryTracking.
4. If stable → suspect CPython or C.
5. Replace malloc with tcmalloc/jemalloc.
6. Generate heap profile.
7. Identify largest allocation stack.
8. Audit ownership contracts.
9. Fix lifecycle mismatch.
10. Re-profile to confirm flat memory curve.

---

# 1️⃣2️⃣ Mixed Runtime Crash Investigation

If crash occurs:

- Inspect hs_err_pid log
- Inspect native stack frames
- Use gdb:

    gdb java core.<pid>
    bt

If libpython appears in stack:

- Investigate refcount
- Investigate GIL correctness
- Inspect Py_DECREF usage

---

# 1️⃣3️⃣ Production Hardening for Java + CPython

Recommended practices:

- Initialize CPython once per JVM lifecycle
- Never reinitialize after Py_FinalizeEx
- Use single interpreter unless strong isolation needed
- Always acquire GIL before Python calls
- Wrap PyObject in deterministic lifecycle (AutoCloseable pattern)
- Stress-test with long-running memory observation
- Monitor RSS, not only heap

---

# 1️⃣4️⃣ Core Insight

In hybrid systems:

Java GC sees only Java heap.  
CPython GC sees only PyObjects.  
malloc sees everything else.

Leaks are almost always:

- Ownership ambiguity
- Missing DECREF
- Missing free()
- Cross-boundary lifecycle mismatch

Do not guess.

Instrument malloc.  
Profile allocations.  
Audit ownership contracts.  
Verify with repeatable measurements.

---

# Final Engineering Principle

When embedding CPython and C libraries inside JVM:

You are building a multi-runtime native system.

Treat memory management as systems engineering —  
not application-level debugging.
