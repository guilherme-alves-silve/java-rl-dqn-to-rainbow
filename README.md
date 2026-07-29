# Implementation of Rainbow-DQN (and it's derivatives) in Java 

## My articles
- [How to Never Forget Deep Q-Networks: Memory Palaces Meet Reinforcement Learning](https://neuralpalace.substack.com/p/how-to-never-forget-deep-q-networks)
- [Connecting Java Reinforcement Learning to Python Gymnasium](https://guilhermealvessilveira.substack.com/p/connecting-java-reinforcement-learning)

## Codes and books used as reference

- [Rainbow is All you need](https://github.com/Curt-Park/rainbow-is-all-you-need/)
  - Main author: Jinwoo Park (Curt)
- [DQN Adventure: from Zero to State of the Art](https://github.com/higgsfield/RL-Adventure/)
  - Main author: Yerzat Dulat (higgsfield)
- [Deep Reinforcement Learning Hands-On, 3rd Edition](https://www.packtpub.com/en-us/product/deep-reinforcement-learning-hands-on-9781835882719)
  - Main author: Maxim Lapan
  - [Github](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-Third-Edition/tree/main/Chapter08)
- [Modern Computer Vision with PyTorch, 2nd Edition](https://www.packtpub.com/en-us/product/modern-computer-vision-with-pytorch-9781803240930)
  - Main authors: V Kishore Ayyadevara, Yeshwanth Reddy
  - [Github](https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch-2E/blob/main/Chapter14/)

## Overview

This project bridges the Java ecosystem and Python's [Gymnasium](https://gymnasium.farama.org/index.html) library to enable Reinforcement Learning training from Java. The core motivation is testability: CARLA is a large and complex simulation environment, so simpler environments like CartPole and Mountain Car are used to validate the integration and shorten feedback cycles.

The series progresses in three articles:

1. **Part 1** — Gymnasium integration via JavaCPP
2. **Part 2 (WIP)** — Incremental algorithm upgrades from DQN to full Rainbow DQN
- [x] DQN
- [x] Double DQN
- [x] PER
- [x] Dueling DQN
- [x] n-Step
- [x] Noisy Networks
- [ ] C51
- [ ] Rainbow
3. **Part 3** — CARLA autonomous driving integration with Rainbow DQN

### Related articles:
- [Video Games and Reinforcement Learning](https://guilhermealvessilveira.substack.com/p/video-games-and-reinforcement-learning)
- [How to Never Forget Deep Q-Networks: Memory Palaces Meet Reinforcement Learning](https://guilhermealvessilveira.substack.com/p/how-to-never-forget-deep-q-networks)

### Related courses and projects:
- [Rainbow is all you need](https://github.com/Curt-Park/rainbow-is-all-you-need)
- [Advanced Reinforcement Learning in Python: cutting-edge DQNs](https://www.udemy.com/course/advanced-deep-qnetworks)

### Related Papers (same as "Rainbow is all you need" project)

01. [V. Mnih et al., "Human-level control through deep reinforcement learning." Nature, 518 (7540):529–533, 2015.](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
02. [van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning." arXiv preprint arXiv:1509.06461, 2015.](https://arxiv.org/pdf/1509.06461.pdf)
03. [T. Schaul et al., "Prioritized Experience Replay." arXiv preprint arXiv:1511.05952, 2015.](https://arxiv.org/pdf/1511.05952.pdf)
04. [Z. Wang et al., "Dueling Network Architectures for Deep Reinforcement Learning." arXiv preprint arXiv:1511.06581, 2015.](https://arxiv.org/pdf/1511.06581.pdf)
05. [M. Fortunato et al., "Noisy Networks for Exploration." arXiv preprint arXiv:1706.10295, 2017.](https://arxiv.org/pdf/1706.10295.pdf)
06. [M. G. Bellemare et al., "A Distributional Perspective on Reinforcement Learning." arXiv preprint arXiv:1707.06887, 2017.](https://arxiv.org/pdf/1707.06887.pdf)
07. [R. S. Sutton, "Learning to predict by the methods of temporal differences." Machine learning, 3(1):9–44, 1988.](http://incompleteideas.net/papers/sutton-88-with-erratum.pdf)
08. [M. Hessel et al., "Rainbow: Combining Improvements in Deep Reinforcement Learning." arXiv preprint arXiv:1710.02298, 2017.](https://arxiv.org/pdf/1710.02298.pdf)

## Technologies

- [Java 25+](https://openjdk.org/)
- [Python 3.12+](https://www.python.org/)
- [Gymnasium](https://gymnasium.farama.org/index.html) — game environments for reinforcement learning
- [JavaCPP](https://github.com/bytedeco/javacpp-presets) — Java↔Python communication via CPython API
- [DJL (Deep Java Library)](https://djl.ai/) — tensor management and deep learning utilities
- [uv](https://github.com/astral-sh/uv) — Python version and virtual environment management

## Integration Design

### Why JavaCPP over Alternatives

Two main integration paths were evaluated:

- **ZeroMQ / Py4J** — Py4J uses a gateway (broker) translating calls between languages; the strict Request→Response ordering of ZeroMQ's socket protocol is fragile and hard to recover from ordering violations.
- **FFM API (Foreign Function and Memory API)** — Requires creating a C-API layer first, then generating bindings via `jextract` for any non-pure-C library.

**JavaCPP** was chosen because it ships a prebuilt [CPython preset](https://github.com/bytedeco/javacpp-presets/tree/master/cpython) and integrates directly with C++ code, making it a natural fit for future CARLA integration as well.

### Embedded Python Lifecycle

The embedded CPython interpreter is initialized once with `Py_Initialize()` and intentionally never finalized with `Py_FinalizeEx()`, because calling it once external modules like NumPy or PyGame have been loaded reliably causes a segmentation fault.

All multi-threaded access to Python objects must be protected with `PyGILState_Ensure()` / `PyGILState_Release()` (wrapped in the `insideGil()` helper), otherwise concurrent reference count mutations can cause double-free crashes and JVM crashes.

### Python Memory Management Reference

The CPython API distinguishes three reference ownership semantics — these must be respected to avoid segfaults:

**New references** (caller owns, must decrement):
`PyObject_CallObject`, `PyObject_Call`, `PyObject_GetAttrString`, `PyImport_AddModule`, `PyUnicode_FromString`, `PyLong_FromLong`, `PyFloat_FromDouble`, `PyBool_FromLong`, `PyBytes_FromStringAndSize`, `PyByteArray_FromStringAndSize`, `PyDict_New`, `PyTuple_New`, `PyList_New`, `PyObject_Str`, `PyUnicode_AsUTF8String`, `PyDict_Items`

**Borrowed references** (do NOT decrement):
`PyTuple_GetItem`, `PyList_GetItem`, `PyDict_GetItem`, `PyModule_GetDict`, `PyErr_Occurred`

**Stolen references** (API takes ownership from caller):
`PyTuple_SetItem`, `PyList_SetItem`

### NumPy Zero-Copy Integration

`NumPyBufferView` implements the [Python Buffer Protocol](https://docs.python.org/3.12/c-api/buffer.html) to expose a NumPy `ndarray`'s contiguous memory directly as a Java `ByteBuffer` — without copying. Changes written to the `ByteBuffer` are reflected in the Python object. The view is deallocated via `PyBuffer_Release()` on `close()`; the original `PyObject` is managed separately.

The `fillFromNumpy()` helper makes a defensive copy for cases where lifecycle management is not yet in place (e.g., no Object Pool), avoiding incorrect deallocation.

Endianness alignment between Java and NumPy is validated once at static initializer time.

### Gymnasium API

The `Gym` class mirrors the `import gymnasium as gym` convention. Environments are constructed via `EnvBuilder`, which generates the Python environment script as a `String` executed inside the embedded interpreter. Key features:

- **Wrappers** — `MaxAndSkipObservation`, `GrayscaleObservation`, `ResizeObservation`, `FrameStackObservation`, and others can be chained via `add()`
- **`PyMap`** — type-safe builder for Python dict parameters
- **`varEnvCode`** — UUID-scoped global variable names prevent collisions in the Python interpreter's `globals` dict
- **`IEnv`** — interface providing `reset()`, `step()`, `render()`, `actionSpaceSample()`, and `close()`, backed by the `Env` implementation

## Optimization

TODO:

- Use `allocateDirect` with `ByteBuffer` to optimize performance, and manage lifecycle together with DJL to reduce memory copies, using an `ObjectPool` of `DirectByteBuffer`.
- Use ASM and ByteBuddy to manage reference counting of Python objects in Java.
- Explore `Py_NewInterpreterFromConfig()` (Python 3.12+) for per-interpreter GIL to improve multiprocessing utilization.
- Evaluate Gymnasium's [vectorization API](https://gymnasium.farama.org/api/vector/) for parallel environment execution.

## Adding a new component

Use this checklist when adding a new network, buffer, or agent.

- [ ] **Managers are named.** Every `newSubManager()` becomes
  `subMgr(parent, "descriptive-name")`.
- [ ] **Models are named.** Every `Model.newInstance(name, device)`
  becomes `newModel(getClass(), device)`.
- [ ] **AutoCloseable.** The component implements `AutoCloseable` and
  its `close()` closes every sub-manager it owns.
- [ ] **Scoping is explicit.** Any `NDArray` used only inside a block
  is created in a `@Cleanup var sub = subMgr(...)` scope.
- [ ] **Parameters are `tempAttach`-ed.** Long-lived parameters used in
  a scoped computation are `tempAttach`-ed to the scope's sub.
- [ ] **No permanent `attach` to the model manager.** Forward outputs
  and intermediates live in scoped subs, not in the model manager.
- [ ] **`@Cleanup` on forward outputs.** `INetwork.forward` callers
  declare `out` with `@Cleanup`.

## CARLA Integration

Resources for Part 3 of the series:

1. [CARLA Simulator - Python API (Basic)](https://blog.wuhanstudio.uk/blog/carla-tutorial-basic/)
2. [CARLA Simulator - Python API (Intermediate)](https://blog.wuhanstudio.uk/blog/carla-tutorial-intermediate/)
3. [CARLA Simulator - Python API (Advanced)](https://blog.wuhanstudio.uk/blog/carla-tutorial-advanced/)
4. [The CARLA Coordinate System](https://blog.wuhanstudio.uk/blog/carla-coordinate/)

## Extras

Project generated with Maven archetype:

```bash
mvn archetype:generate -DarchetypeGroupId=org.apache.maven.archetypes \
    -DarchetypeArtifactId=maven-archetype-quickstart \
    -DarchetypeVersion=1.5 \
    -DgroupId=br.com.guialves.rflr.dqn \
    -DartifactId=java-rl-dqn-to-rainbow \
    -DinteractiveMode=false
```

### Python Environment Setup

A `gymnasium/` folder at the project root contains:

```
README.md         # uv install instructions + required env vars
requirements.txt  # pinned Python dependencies
```

Key environment variable required by Java:

```bash
export JAVA_RL_SITE_PACKAGES=/path/to/java-rl-dqn-to-rainbow/gymnasium/.venv/include/site/python3.12
```

`requirements.txt`:

```
numpy==2.3.5
pygame==2.6.1
matplotlib==3.10.8
gymnasium[classic-control,box2d]==1.2.2
```

---

## AI Usage

### Process

1. **Human defines the contract** — The author writes a reference implementation and defines the public interface (methods, signatures, and behavior).
2. **AI generates alternatives** — AI tools (closed-source and open-source) produce alternative implementations (e.g., converting pointer-based structures to flat arrays) that **must respect the existing interface and signatures**.
3. **Human validates** — The author reviews, tests, and decides whether to accept, modify, or reject each AI-generated variant.
4. **AI generates scripts** - Scripts not used in the main flow, such as `per_evolution_distribution_time.py`, are almost entirely generated by AI, but are also reviewed by the author.
5. **AI generates documentation** - Documentation is almost entirely generated by AI, but is also reviewed by the author.

### Example

- **Human writes**: `SumSegmentTree` (pointer-based) with methods `update()` and `sampleIndexByValueInRange()`
- **AI generates**: `SumSegmentTreeFlat` (flat array-based) with the same interface
- **Human ensures**: Correctness, performance, and style consistency

### Accountability

The author is **fully responsible** for all code — including AI-generated variants. AI is a tool for generating implementation alternatives under human guidance, not an autonomous author.

---

## JVM GC Configuration Reference

### 1. Memory Sizing & Containers

- `-Xms4g -Xmx4g` — Fixed heap; prevents resizing jitter but disables uncommitting RAM.
- `-XX:MaxDirectMemorySize=2g` — Caps off-heap (NIO) memory. **Crucial for JNI/CPython.**
- `-XX:+UseContainerSupport` — Enables awareness of Docker/K8s memory limits.
- `-XX:MaxRAMPercentage=75.0` — Sets heap as % of total container RAM.

### 2. Collector Strategies (Pick ONE)

- **G1 GC (Balanced):** `-XX:+UseG1GC -XX:MaxGCPauseMillis=50`
  Best for general purpose. Use `-XX:+ExplicitGCInvokesConcurrent` to prevent STW on `System.gc()`.
- **ZGC (Low Latency):** `-XX:+UseZGC -XX:ZUncommitDelay=30`
  Sub-10ms pauses. Requires `-Xms < -Xmx` to return memory to OS.
- **Shenandoah (Concurrent):** `-XX:+UseShenandoahGC`
  Ultra-low latency; evacuates heap concurrently with application threads.
- **Serial (Small Apps):** `-XX:+UseSerialGC`
  Use only for micro-containers (<512MB) or CLI tools.

### 3. Dynamic Scaling (G1/Serial)

- `-XX:MinHeapFreeRatio=5` — Min free space before heap expansion.
- `-XX:MaxHeapFreeRatio=20` — Max free space before returning RAM to OS.

### ⚠️ Hybrid System Tip (Java + CPython)

Avoid `MaxRAMPercentage=75`. Use **50.0** instead to ensure the Python runtime and native C libraries have sufficient headroom to avoid OS-level OOM kills.
