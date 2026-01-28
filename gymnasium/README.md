# [How to Never Forget Deep Q-Networks: Memory Palaces Meet Reinforcement Learning](https://guilhermealvessilveira.substack.com/p/how-to-never-forget-deep-q-networks)

```
pip install uv
uv venv --python 3.12
.venv/Scripts/activate
uv pip install -r requirements.txt 
```

The used environment is the [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

Pseudo code of env_proxy executor

inside env_proxy.render(), env_proxy.sample_action() etc, the python env is
executed, and inside this method the env_proxy send the result to the Java server.

while True:

    command = receive_command()
    switch (command):
        case "1" # RENDER
            env_proxy.render()
        case "2" # SAMPLE ACTION
            env_proxy.sample_action() # inside this you sample the action and return the result,
            # we can create a sample_action_in_step to sample_action and then invoke the env.step and return the result
            # to the Java Server
        case "3" # STEP
            env_proxy.step()
        ...
