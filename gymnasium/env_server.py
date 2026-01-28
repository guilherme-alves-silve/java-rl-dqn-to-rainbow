# https://gymnasium.farama.org/environments/classic_control/cart_pole/ (default: \"CartPole-v1\")
import zmq
import json
import argparse

from env_proxy import EnvironmentProxy, EnvOperations
from utils import create_socket


if __name__ == "__main__":

    args = argparse.ArgumentParser(description="Gymnasium server with ZeroMQ")
    args.add_argument("--port", type=int, default=5555,
                      help="Port to bind the ZeroMQ REP socket (default: 5555)")
    args.add_argument("--timeout", type=int, default=5000,
                      help="Timeout of the server in milliseconds (default: 5000)")
    args.add_argument("--env_name", type=str, help="Environment for the agent to train-on")
    args.add_argument("--env_params", type=json.loads, default="{}",
                      help="Environment custom parameters, must be passed in the JSON format")
    args.add_argument("--env_episodes", type=int, help="Environment episodes to run")

    context = zmq.Context()
    socket = create_socket(context, args.port, args.timeout)
    env_proxy = EnvironmentProxy(args, context, socket)

    print(f"[+] Server listening on port {args.port}")
    try:
        while True:
            res = socket.recv_string()
            if EnvOperations.ACTION_SPACE_STR.value == res:
                env_proxy.action_space_sample()

    except Exception as ex:
        print(f"Error: {ex}")
    finally:
        env_proxy.close()
        socket.close()
        context.term()
