# Java integration with Stable-Retro

- [stable-retro](https://stable-retro.farama.org/)

This project just work with Ubuntu/WSL2. The library stable-retro only works with Linux and MacOS.
- To make this work in Windows, is necessary more dedication, but seens possible.

```
sudo apt update
sudo apt install -y software-properties-common

sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update

sudo apt install -y libgl1 libglu1-mesa libegl1 mesa-utils
# You can see the output of stable-retro without errors when rendering

sudo apt install -y python3.12 python3.12-venv python3.12-dev

sudo apt install -y openjdk-25-jdk
sudo update-alternatives --config java
# Select Java 25, this project uses Java 25

cd ~/java-rl-dqn-to-rainbow/stable-retro

python3.12 -m venv .venv
which python
# Example output: /home/<YOUR_USERNAME>/java-rl-dqn-to-rainbow/stable-retro/.venv/bin/python
source .venv/bin/activate
pip install -r requirements.txt

python -m stable_retro.import /home/<YOUR_USERNAME>/roms/
# Use your legal roms

python smoke_test.py

export JAVA_RL_SITE_PACKAGES=/home/<YOUR_USERNAME>/java-rl-dqn-to-rainbow/gymnasium/.venv/Lib/site-packages
```

## Windows + WSL2: open the project in VS Code

1. Install the [WSL extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-wsl) in VS Code.
2. Open a WSL terminal, go to the project folder and run:

```bash
   code .
```

3. Check that the bottom-left corner shows `WSL: Ubuntu`.

Tips:

- Keep the project in `/home/...`, not `/mnt/c/...` (much faster).
- Install Git inside WSL: `sudo apt install git`.
- Install language extensions in the WSL window (**Install in WSL** button).
