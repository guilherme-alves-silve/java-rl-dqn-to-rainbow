# [How to Never Forget Deep Q-Networks: Memory Palaces Meet Reinforcement Learning](https://guilhermealvessilveira.substack.com/p/how-to-never-forget-deep-q-networks)

## Technologies

- [Java 25+](https://openjdk.org/)
- [Python 3.12+](https://www.python.org/)
- [gymnasium](https://gymnasium.farama.org/index.html)
  - For the game environment used for reinforcement learning
- [javacpp](https://github.com/bytedeco/javacpp-presets)
  - To make the communication between Java and Python

# Build the gymnasium environment adapted to Java

`docker build -t gymnasium4j -f Dockerfile .`

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