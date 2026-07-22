# [How to Never Forget Deep Q-Networks: Memory Palaces Meet Reinforcement Learning](https://guilhermealvessilveira.substack.com/p/how-to-never-forget-deep-q-networks)

```
pip install uv
uv python install cpython-3.12.1-windows-x86_64-none
uv venv --python 3.12.1
.venv/Scripts/activate
uv pip install -r requirements.txt 
```

The used environment is the [CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/)

Configure the environment varibles. Example:

- `export JAVA_RL_SITE_PACKAGES=/path/to/java-rl-dqn-to-rainbow/gymnasium/.venv/include/site/python3.12`

Sometimes the python don't add the Include, do the process below:

Windows (Git Bash):
- `cp -r "/c/Users/YOUR_USER_NAME/AppData/Local/Python/pythoncore-3.12-64/Include/"* "venv/Include/"`

Linux:
- `cp -r /usr/include/python3.12/* venv/include`
