

## Technologies

- [Java 25+]()
- [Python 3.12+]()
- [gymnasium]()
  - For the game environment used for reinforcement learning
- [zeromq](https://zeromq.org/)
  - For zero copy between environments


Project generated with maven archtype:

```
mvn archetype:generate -DarchetypeGroupId=org.apache.maven.archetypes \
    -DarchetypeArtifactId=maven-archetype-quickstart \
    -DarchetypeVersion=1.5 \
    -DgroupId=br.com.guialves.rflr.dqn \
    -DartifactId=java-rl-dqn-to-rainbow \
    -DinteractiveMode=false
```