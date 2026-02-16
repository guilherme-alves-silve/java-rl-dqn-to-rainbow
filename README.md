# [How to Never Forget Deep Q-Networks: Memory Palaces Meet Reinforcement Learning](https://guilhermealvessilveira.substack.com/p/how-to-never-forget-deep-q-networks)

## Technologies

- [Java 25+](https://openjdk.org/)
- [Python 3.12+](https://www.python.org/)
- [gymnasium](https://gymnasium.farama.org/index.html)
  - For the game environment used for reinforcement learning
- [javacpp](https://github.com/bytedeco/javacpp-presets)
  - To make the communication between Java and Python

## Optimization

TODO: Use allocateDirect with ByteBuffer to optimize performance, and manage
lifecycle together with DJL to reduce as much as possible memory copy, using ObjectPool of DirectByteBuffer.

## Extras

Project generated with maven archtype:

```
mvn archetype:generate -DarchetypeGroupId=org.apache.maven.archetypes \
    -DarchetypeArtifactId=maven-archetype-quickstart \
    -DarchetypeVersion=1.5 \
    -DgroupId=br.com.guialves.rflr.dqn \
    -DartifactId=java-rl-dqn-to-rainbow \
    -DinteractiveMode=false
```

## JVM GC Configuration Reference

### 1. Memory Sizing & Containers
* `-Xms4g -Xmx4g`: Sets fixed heap; prevents resizing jitter but disables uncommitting RAM.
* `-XX:MaxDirectMemorySize=2g`: Caps off-heap (NIO) memory. *Crucial for JNI/CPython.*
* `-XX:+UseContainerSupport`: Enables awareness of Docker/K8s memory limits.
* `-XX:MaxRAMPercentage=75.0`: Sets heap as % of total container RAM.

### 2. Collector Strategies (Pick ONE)
* **G1 GC (Balanced):** `-XX:+UseG1GC -XX:MaxGCPauseMillis=50`
  * Best for general purpose. Use `-XX:+ExplicitGCInvokesConcurrent` to prevent STW on `System.gc()`.
* **ZGC (Low Latency):** `-XX:+UseZGC -XX:ZUncommitDelay=30`
  * Sub-10ms pauses. Requires `-Xms < -Xmx` to return memory to OS.
* **Shenandoah (Concurrent):** `-XX:+UseShenandoahGC`
  * Ultra-low latency; evacuates heap concurrently with application threads.
* **Serial (Small Apps):** `-XX:+UseSerialGC`
  * Use only for micro-containers (<512MB) or CLI tools.

### 3. Dynamic Scaling (G1/Serial)
* `-XX:MinHeapFreeRatio=5`: Min free space before heap expansion.
* `-XX:MaxHeapFreeRatio=20`: Max free space before returning RAM to OS.

### ⚠️ Hybrid System Tip (Java + CPython)
Avoid `MaxRAMPercentage=75`. Use **50.0** instead to ensure the Python runtime and Native C libraries have sufficient headroom to avoid OS-level OOM kills.
