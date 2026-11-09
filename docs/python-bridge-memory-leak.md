# Python Bridge Reference Leaks (CPython refcount)

> Companion to [`memory-management.md`](memory-management.md), which covers the
> **DJL** side of the tree (`NDManager` / `NDArray`). This document covers the
> **CPython** side: the reference counting of `PyObject` across the Java ↔ Python
> bridge.

## TL;DR

`org.bytedeco.cpython.PyObject` is declared `@Opaque` in the JavaCPP binding and
**has no deallocator**. Therefore, on a raw `PyObject`:

| JavaCPP API | What it does on a `PyObject` |
|---|---|
| `Pointer.close()` | **no-op** — never calls `Py_DECREF`, never invalidates the pointer |
| `Pointer.retainReference()` | **no-op** — there is no `ReferenceCounter` to retain |
| Lombok `@Cleanup` on a `PyObject` | **no-op** — it calls `close()` |

Every `@Cleanup` / try-with-resources over a **raw** `PyObject` silently leaks one
Python reference. Combined with the fact that the leaked objects are the
observation tensors returned by `env.step()`, this produced a **native
(out-of-heap) leak of ~28.9 KB per step** in `AgentCNNDQNMain`, ending in an
`Out of Memory Error` from the CPython allocator (`arena.cpp:186`) after a few
hundred thousand frames.

**The fix in this project is `PyRef`** — a thin `PyObject` subclass that carries
the ownership and releases the reference on `close()`:

```java
try (var ref = PyRef.of(PyObject_GetAttrString(obj, "attr"))) {
    // the reference is released here, exactly once
}
```

`PyRef` is used by `PythonRuntime.eval/callObj/callMethod/callFunction` and by the
`PythonDataStructures` builders. It works around the missing deallocator without
touching the upstream binding.

## Symptom

`AgentCNNDQNMain` (Atari, CNN DQN) grew ~29 KB of native memory per frame and
died with:

```
# Native memory: malloc  ->  Out of Memory Error (arena.cpp:186)
# num_arena_births: 124
# WorkingSet: 77.590 MB   (on a 95.800 MB machine)
```

The DJL manager tree was **clean** — the leak was not in the `NDManager`
hierarchy. It lived entirely in the CPython heap.

## Root cause

### 1. `PyObject.close()` does not decrement the refcount

`org/bytedeco/cpython/PyObject.java` (javacpp 1.5.10):

```java
@Opaque @Properties(inherit = org.bytedeco.cpython.presets.python.class)
public class PyObject extends Pointer {
    public PyObject() { super((Pointer)null); }
    public PyObject(Pointer p) { super(p); }
}
```

No deallocator is declared. In `org/bytedeco/javacpp/Pointer.java`:

```java
@Override public void close() {          // line 766
    releaseReference();
}

public <P extends Pointer> P retainReference() {   // line 803
    ReferenceCounter r = (ReferenceCounter)deallocator;
    if (r != null) r.retain();
    return (P)this;
}

public boolean releaseReference() {     // line 816
    DeallocatorReference r = (DeallocatorReference)deallocator;
    if (r != null && r.release()) { ... }
    return false;                       // <- no deallocator: nothing happens
}
```

Both `close()` and `retainReference()` are no-ops for `PyObject`.

**Proof (probe `DeallocProbe`)** — the `deallocator` field of `Pointer` (line 606)
read by reflection at runtime:

```
PyObject    (@Opaque, no deallocate()) -> null
Py_buffer  (not @Opaque, with allocate()) -> NativeDeallocator[ownerAddress=0x...,deallocatorAddress=0x...]

sys.getrefcount before close()  = 3
sys.getrefcount after close()   = 3      <- close() does not decref
isNull() after close()          = false   <- the pointer is still valid
address after close()           = 2495397347248  <- intact

refcount before = 3, after = 3   -> retainReference() did NOT change the refcount (no-op)
```

And the control case, a class that *does* have a deallocator:

```
Py_buffer.close() -> address = 0   (actually deallocated)
```

So `close()` on a `PyObject` is provably a no-op, and `retainReference()` is
provably a no-op.

### 2. Which APIs hand a *new* reference to Java

Every one of these returns a **new** reference that Java now owns and must
release with an explicit `Py_DECREF`:

| C API | Java wrapper | Released in |
|---|---|---|
| `PyObject_GetAttrString` | `PythonRuntime.attr` | caller must release it |
| `PyObject_Str` | `PythonRuntime.toStr` | `PythonRuntime.java:232` |
| `PyUnicode_AsUTF8String` | `PythonRuntime.toStr` | `PythonRuntime.java:235` |
| `PyObject_Type` | `PythonRuntime.toType` | `PythonRuntime.java:190` |
| `PyObject_CallObject` | `PythonRuntime.callObj` / `callFunction` / `callMethod` | wrapped in a `PyRef` |
| `PyRun_StringFlags` | `PythonRuntime.eval` | wrapped in a `PyRef` |
| `PyDict_Items` | `PythonRuntime.printDict` | `PythonRuntime.java:529` |

These return a **borrowed** reference and must **not** be released:

| C API | Java wrapper |
|---|---|
| `PyTuple_GetItem` / `PyList_GetItem` | `PythonDataStructures.getItem` (used as `getItem(result, 0)`) |
| `PySequence_GetItem` | `PythonDataStructures.getSequenceItem` (it increfs first) |
| `PyBytes_AsString` | `PythonRuntime.toStr` (inner block) |
| `PyErr_Occurred` | `PythonRuntime.checkError` |
| `PyBool_Type`, `PyLong_Type`, ... | `PythonTypeChecks.*` (static type objects) |
| `PyDict_Next` (key/value) | `PythonDataStructures.toMap` |

### 3. The original bug

`PythonRuntime.callFunction` ended with `result.retainReference()`, which reads
as "the reference is already retained, `close()` will release it". That is
**false**: `retainReference()` is a no-op, and `close()` is a no-op. The
returned tuple — and the observation array inside it — stayed alive forever.

```java
// BEFORE - in Env.step()
@Cleanup var result = callFunction(pyStep, action.pyObj);   // never released
```

The first fix was an explicit `Py_DECREF` in a `finally` block:

```java
// AFTER - in Env.step()
var result = callFunction(pyStep, action.pyObj);
try {
    ...
} finally {
    decRef(result);
}
```

That is correct but noisy: every call site has to remember the `finally` block,
and forgetting one is a silent leak. `PyRef` removes the need to remember.

## The fix in this project: `PyRef`

`src/main/java/br/com/guialves/rflr/python/PyRef.java`:

```java
public final class PyRef extends PyObject implements AutoCloseable {

    private final boolean useGil;

    private PyRef(boolean useGil, Pointer owned) {
        super(owned);          // Pointer copy constructor == reinterpret_cast
        this.useGil = useGil;
    }

    /** Takes ownership of a Python reference, so that it gets released on close(). */
    public static PyRef of(PyObject owned) {
        return new PyRef(false, owned);
    }

    /** Same as {@link #of(PyObject)}, but acquires the GIL around the release. */
    public static PyRef ofGIL(PyObject owned) {
        return new PyRef(true, owned);
    }

    public PyObject get() {
        return this;
    }

    /** The current reference count, as reported by Py_REFCNT(). */
    public long refCount() {
        return isNull() ? -1 : Py_REFCNT(this);
    }

    @Override
    public void close() {
        if (isNull()) {
            return;             // idempotent: a second close() does nothing
        }

        var self = this;
        if (useGil) {
            int state = PyGILState_Ensure();
            try {
                Py_DECREF(self);
            } finally {
                PyGILState_Release(state);
            }
        } else {
            Py_DECREF(self);
        }
        setNull();              // the pointer is invalidated
    }
}
```

### Design notes

- **It extends `PyObject`** so that it can be passed anywhere a `PyObject` is
  expected, without touching a single call site of the existing API.
- **The `Pointer` copy constructor acts as a `reinterpret_cast`.** It copies the
  address and nothing else. Because `PyObject` has no deallocator, no
  `ProxyDeallocator` is created — the `PyRef` does not add a second owner.
- **`close()` is idempotent.** It checks `isNull()` first and calls `setNull()`
  after the release, so a double `close()` (or a `close()` after an explicit
  `Py_DECREF`) cannot corrupt the counter.
- **`ofGIL()` takes the GIL around the release.** `Py_DECREF` may run the
  object's deallocator, which can execute arbitrary Python code; that code must
  hold the GIL. Use it when the reference may be released from a thread that
  does not already hold it. `of()` does not, and must only be used from a thread
  that already holds the GIL.
- **`refCount()` exposes `Py_REFCNT`**, which is what makes the leak measurable
  without going through Python.

### Where it is used

| Site | Change |
|---|---|
| `PythonRuntime.eval` (`PythonRuntime.java:171`) | returns `PyRef.of(result)` |
| `PythonRuntime.callObj` (`PythonRuntime.java:298`) | returns `PyRef.of(result)` |
| `PythonRuntime.callMethod` (`PythonRuntime.java:343`) | returns `PyRef.of(result)` |
| `PythonRuntime.callFunction` (`PythonRuntime.java:377`) | returns `PyRef.of(result)` |
| `PythonDataStructures.pyList/pyTuple/pySet/pyDict` | the builders return `PyRef.of(...)` |

### The static-type subtlety (read this before changing a return type)

`PythonRuntime.eval`, `callObj`, `callMethod` and `callFunction` are declared
as returning **`PyObject`**, but they actually return a **`PyRef` instance**.
`Env.step()` therefore does:

```java
@Cleanup var result = callFunction(pyStep, action.pyObj);
```

Lombok generates `result.close()` in a `finally` block. `result` is statically
typed as `PyObject`, but `close()` is virtual and the runtime type is `PyRef`,
so **`PyRef.close()` runs and the reference is released**.

This works today, but it is a **fragile** contract:

- If any of those four methods ever returns a plain `PyObject` again, every
  `@Cleanup` over it silently becomes a no-op and the leak comes back — with no
  compile error and no test failure until the native OOM.
- The robust form is to declare the return type as `PyRef`:

  ```java
  public static PyRef callFunction(PyObject fn, PyObject... args) { ... }
  ```

  `PyRef` *is* a `PyObject`, so every existing caller keeps compiling.

## Ownership rules

The CPython C API has **four** ownership categories, not three. Getting the
fourth one wrong is what turns a leak into a crash.

### New reference — you own it, release it

```
PyObject_GetAttrString   PyObject_CallObject      PyObject_Str
PyObject_Type            PyUnicode_AsUTF8String   PyRun_StringFlags
PyDict_Items/Keys/Values PySequence_GetItem       PyList_GetItemRef
PyDict_GetItemRef        PyObject_GetIter         PyList_New / PyDict_New
```

Release exactly once, with `Py_DECREF` or by wrapping it in a `PyRef`.

### Borrowed reference — Python owns it, never release it

```
PyDict_GetItem           PyList_GetItem           PyTuple_GetItem
PyErr_Occurred           PyModule_GetDict         PyImport_AddModule
PyEval_GetGlobals        PyType_GetDict           PyCell_Get
PyStructSequence_GetItem PyDict_SetDefault
```

A `Py_DECREF` on a borrowed reference is a **use-after-free**, and the symptom
(a crash somewhere random) is much worse than a leak.

### Immortal / static object — never release it

```
PyExc_TypeError, PyExc_ValueError, ... (all PyExc_*)
PyBool_Type, PyLong_Type, PyDict_Type, ... (all Py*_Type)
Py_None, Py_True, Py_False, Py_Ellipsis, Py_NotImplemented
```

These are pre-allocated singletons. In CPython 3.12+ the type objects are
immortal, and the `PyExc_*` accessors do not incref either.

### Steals — the function takes ownership, do NOT release afterwards

```
PyList_SetItem       PyTuple_SetItem        PyStructSequence_SetItem
PyList_SET_ITEM      PyTuple_SET_ITEM       (macros)
PyModule_AddObject   (steals only on success)
PyModule_Add         (3.13+, steals even on error)
```

`PyList_SetItem` and `PyTuple_SetItem` steal **even when they fail**, so the
reference must not be released on either path. `PyModule_AddObject` is the
worst case: the success path forbids the release and the error path requires
it.

Conversely, the setters that do **not** steal — and therefore require you to
release your own reference afterwards — are:

```
PyDict_SetItem       PyDict_SetItemString   PyDict_SetDefault
PyList_Append        PyList_Insert          PySet_Add
PyObject_SetAttr     PyObject_SetItem       PyModule_AddObjectRef
```

`PyDict_SetItem` is the important one: the dictionary stores its **own**
reference, and yours remains yours. Releasing it afterwards is correct and
required:

```java
try (PyObject key = ...; PyObject value = ...) {
    if (python.PyDict_SetItem(dict, key, value) != 0) {
        throw new IllegalStateException("PyDict_SetItem failed");
    }
    // key and value are released here: correct on both paths.
    // The dict holds its own reference, so the object stays alive.
}
```

Not releasing them after `PyDict_SetItem` leaks one reference per insertion.

## Measured impact

`PyRefProbe` measures `sys.getallocatedblocks()` (live Python objects) per call,
2000 iterations, before and after the fix:

| Call site | Before | After |
|---|---|---|
| `attr(obj, 'shape')` without release | +1.2115 blocks/call | +1.2115 (expected: the leak) |
| `attr(obj, 'shape')` with `decRef` | +0.0000 | +0.0000 |
| `toStr(numpy array)` | +2.8950 | +0.8950 * |
| `toStr(bool / str / int)` | — | +0.0000 |
| `attrStr(obj, 'dtype')` | +2.2170 | +0.0035 |
| `toType(obj)` | +2.0000 | +0.0000 |
| `EnvStateMetadata.fromNumpy(arr)` | +3.0060 | +0.0000 |
| `fillFromNumpy(arr, buffer)` | +0.0000 | +0.0000 |
| `toBoolArray([True,False,True])` | — | +0.0000 |
| **`env.reset(manager)`** | **+2.9995** | **+0.0030** |
| **`env.step(action, manager)`** | **+4.0010** | **+0.0010** |
| `env.actionSpaceSample()` + close | — | +0.0000 |

\* The residual `~0.9` blocks/call of `toStr(numpy_array)` is **not ours**:
in pure Python, `str(numpy_array)` leaks the same amount (2006 blocks for 2000
calls, measured with the venv interpreter). It is an internal numpy cache.
`toStr` is not used on arrays in the training loop, so it does not accumulate.

`PyRefTest` measures the release mechanism itself, with `Py_REFCNT`, over 1000
iterations:

| Case | Reference count after the loop |
|---|---|
| plain `PyObject` in try-with-resources | `+1000` (leaked, as expected) |
| `PyRef.of(...)` in try-with-resources | `+0` (released) |
| `PyRef.close()` called twice | `+0` (idempotent) |

## End-to-end validation

`AgentCNNDQNMain.run()`, 40.000 frames, Atari `Box(0, 255, (4, 84, 84), uint8)`:

```
TRAIN: 40000 frames in 56.3 s
  ep   0: commit = 3.15 GB | delta since ep0 =  +0.00 MB | arrays buffer = 426
  ep  10: commit = 3.13 GB | delta since ep0 = -19.40 MB | arrays buffer = 1000
  ep  67: commit = 3.13 GB | delta since ep0 = -18.98 MB | arrays buffer = 1000
  FINAL: total grown = -18.98 MB in 68 episodes
```

The commit charge is **flat** (it even shrinks slightly) and the replay buffer
stabilises at `1000` arrays (`bufferCapacity 500 × 2` per experience). Before
the fix the same loop grew by **28.9 KB per step**.

The final manager tree has exactly the three expected sub-managers:

```json
{"name":"parent-...","totalResources":3,"byType":{"PtNDManager":3},"children":[
  {"name":"clone-DeepQNetworkCNN-...","totalResources":10},
  {"name":"ExperienceReplayBuffer-...","totalResources":1000},
  {"name":"DeepQNetworkCNN-...","totalResources":10}]}
```

## Fixes applied

| File | Change |
|---|---|
| `python/PyRef.java` | **new** — ownership-carrying wrapper: `close()` calls `Py_DECREF` and invalidates the pointer |
| `python/PythonRuntime.java:171` | `eval` returns a `PyRef` |
| `python/PythonRuntime.java:298` | `callObj` returns a `PyRef` |
| `python/PythonRuntime.java:343` | `callMethod` returns a `PyRef` |
| `python/PythonRuntime.java:377` | `callFunction` returns a `PyRef` |
| `python/PythonRuntime.java:190` | `toType` releases the `PyObject_Type` reference |
| `python/PythonRuntime.java:232,235` | `toStr` releases `PyObject_Str` + `PyUnicode_AsUTF8String` |
| `python/PythonRuntime.java:272` | `attrStr` releases the `PyObject_GetAttrString` reference |
| `python/PythonRuntime.java:478` | `decRef(PyObject)` — null-safe `Py_DECREF`, for the sites that cannot use `PyRef` |
| `python/PythonRuntime.java:529` | `printDict` releases the `PyDict_Items` reference |
| `python/PythonDataStructures.java:183,216,222,229,268,274,280,325,356,383` | the list/tuple/set/dict builders return a `PyRef` |
| `gymnasium4j/Env.java:103,132,163,185,204` | `reset`/`step`/`stepVec`/`renderVec`/`render` release the result via `@Cleanup` over the `PyRef` |
| `gymnasium4j/Env.java:242-247` | `close()` releases the environment handles |
| `gymnasium4j/EnvStateMetadata.java:37` | releases the `shape` attribute |
| `python/numpy/NumPyByteBuffer.java:225-230,241-246` | `nbytes` and `nestedAttrStr` release their references |

`ActionSpaceType.detectActionSpaceType` already released its references
explicitly — that is why the leak was *partially* visible before: the code that
had been written with `decRef` in mind was correct, the code written with
`@Cleanup` in mind was not.

## Rules

1. **Never use `@Cleanup` / try-with-resources on a raw `PyObject`.** It does
   nothing. Wrap the reference in a `PyRef`, or release it explicitly with
   `decRef(...)` in a `finally` block.
2. **Prefer `PyRef` over a manual `decRef`** in new code: the ownership travels
   with the value and cannot be forgotten. Use `decRef` only where a `PyRef`
   cannot be returned (for example, a local temporary inside a helper).
3. **`attr()` returns an owned reference.** The caller must release it.
4. **`getItem()` returns a borrowed reference.** Never release it.
5. **Never release a borrowed or immortal reference.** A `Py_DECREF` on a
   borrowed object is a use-after-free, and the symptom (a crash in a random
   place) is much worse than a leak.
6. **Never release a reference after a stealing call** (`PyList_SetItem`,
   `PyTuple_SetItem`, `PyModule_AddObject`). That is a double-free.
7. **Release the GIL-aware variant (`PyRef.ofGIL`) when releasing from a
   thread that does not already hold it.**

## How to measure

### Python objects

```java
var value = eval("__import__('sys').getallocatedblocks()");
try {
    return toLong(value);
} finally {
    decRef(value);
}
```

`sys.getallocatedblocks()` counts live *pymalloc* blocks. Objects larger than
512 bytes go straight to `malloc` and are **not** counted — an ndarray
observation is invisible to this metric, but its *tuple* is not.

For a single reference, `PyRef.refCount()` is more precise: it reads
`Py_REFCNT` directly, with no Python round-trip.

### Native memory

Use the **commit charge**, never the working set:

```java
var os = (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();
long committed = os.getCommittedVirtualMemorySize();
```

The JDK implementation class (`com.sun.management.internal.OperatingSystemImpl`)
is package-private, so the bean must be read **through the interface**. On
JDK 25 `getProcessMemoryInfo()` no longer exists.

**Never measure with `WorkingSetSize`**: Windows trims the working set, and a
process with 1.190 MB committed can report a 52 MB working set (or the
opposite). The commit charge is the only metric that reflects native `malloc`.

## Pitfalls

- **`PyErr_Print()` output is invisible.** The embedded CPython writes to the
  native `stderr`, not to the Java `System.err`. A Python exception raised by
  the bridge shows up only as `RuntimeException: Failed to call function`.
- **`gym.make()` applies `OrderEnforcing`.** Calling `env.step()` before the
  first `env.reset()` raises `ResetNeeded` in Python — and, because of the
  point above, looks like an opaque "Failed to call function" in Java. Always
  `reset()` first.
- **`str(numpy_array)` leaks inside numpy** (~1 block/call). Not our bug, but
  do not call it in a loop.
- **A `PyRef` released without the GIL can crash.** `Py_DECREF` may run a
  Python deallocator. `PyRef.ofGIL` exists for that case.
- **A `PyRef` whose `close()` is skipped is a leak, not a crash.** The pointer
  stays valid; the object is simply never released.

## Regression tests

`src/test/java/br/com/guialves/rflr/gymnasium4j/PythonBridgeMemoryLeakTest.java`
covers three invariants:

| Test | Metric | Budget | Measured |
|---|---|---|---|
| `stepShouldNotLeakPythonReferences` | Python blocks per step | < 0.5 | 0.0010 |
| `resetShouldNotLeakPythonReferences` | Python blocks per reset | < 0.5 | 0.0020 |
| `stepShouldNotLeakNativeMemory` | commit charge bytes per step | < 256 | 0 |

Against the unfixed bridge the same test fails with **7.1315 blocks/reset** and
**4.0010 blocks/step** (plus 526 native bytes/step on CartPole).

```bash
mvn test -Dtest=PythonBridgeMemoryLeakTest
```

`src/test/java/br/com/guialves/rflr/python/PyRefTest.java` covers the release
mechanism itself:

| Test | Asserts |
|---|---|
| `testCloseOnPlainPyObjectDoesNotRelease` | the raw `PyObject` still leaks (documents the upstream behaviour) |
| `testPyRefReleasesTheReference` | `PyRef` releases exactly one reference per iteration |
| `testPyRefCloseIsIdempotent` | a second `close()` does not release twice, and the pointer is invalidated |

```bash
mvn test -Dtest=PyRefTest
```

## Upstream

The root cause is in the binding, not in this project: `PyObject` is `@Opaque`
without a deallocator, so the whole CPython preset cannot release references
through the normal JavaCPP lifecycle.

The fix is being pushed upstream to `bytedeco/javacpp-presets`:

- **PR** adding a `PyRef` helper to the `cpython` preset, so that every
  embedder gets the same ownership-carrying wrapper without writing it.
- A **preset-only** experiment that gives the generated `PyObject` a base class
  with a working `close()`, using `Info.base()` — no core change required.
- An **ownership reference** classifying all ~490 `PyObject`-returning functions
  of the preset into the four categories above (new / borrowed / immortal /
  steals), so that the "does it steal?" question has a documented answer for
  every API.

`PyRef` in this project is the local, immediate version of that idea: the same
contract, without waiting for the upstream release.

## Relation to `TODO.md`

`TODO.md` proposes using Byte Buddy to inject `Py_DECREF` automatically, with
the distinction between owned and borrowed references. `PyRef` is the explicit
version of that distinction, and it is not a substitute for the automated
approach: it still requires the author of each call site to know the ownership
of the value. But it removes the `finally` blocks, and it gives the Byte Buddy
plugin exactly the anchors it needs — the set of places where a reference
crosses the boundary into Java.

The upstream work (see above) attacks the same problem one level lower, at the
binding itself, where the distinction can be encoded once instead of at every
call site.

## Related

- [`memory-management.md`](memory-management.md) — the DJL side (`NDManager`).
- [`extra/DEBUG.md`](extra/DEBUG.md) — native leak debugging methodology.
- [`TODO.md`](TODO.md) — Byte Buddy based automatic `PyObject` cleanup.
