

## Technologies

- [Java 25+]()
- [Python 3.12+]()
- [gymnasium]()
  - For the game environment used for reinforcement learning
- [zeromq](https://zeromq.org/)
  - For zero copy between environments

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