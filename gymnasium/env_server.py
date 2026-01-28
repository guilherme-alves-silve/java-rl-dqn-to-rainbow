# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import zmq
import json
import argparse

from env_proxy import EnvironmentProxy, EnvOperations
from utils import create_socket


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Gymnasium server with ZeroMQ")
    parser.add_argument("--port", type=int, default=5555,
                        help="Port to bind the ZeroMQ REP socket (default: 5555)")
    parser.add_argument("--timeout", type=int, default=5000,
                        help="Timeout of the server in milliseconds (default: 5000)")
    parser.add_argument("--env_name", type=str, default="CartPole-v1",
                        help="Environment for the agent to train-on")
    parser.add_argument("--env_params", type=json.loads, default="{}",
                        help="Environment custom parameters (JSON)")
    parser.add_argument("--env_episodes", type=int, default=1,
                        help="Environment episodes to run")

    args = parser.parse_args()

    context = zmq.Context()
    socket = create_socket(context, args.port, args.timeout)
    env_proxy = EnvironmentProxy(args, context, socket)

    print(f"[+] Server listening on port {args.port}")

    try:
        while True:
            cmd = socket.recv_string()
            op = EnvOperations(cmd)

            # ---- Space ops ----
            if op == EnvOperations.ACTION_SPACE_SAMPLE:
                env_proxy.action_space_sample()

            elif op == EnvOperations.ACTION_SPACE_STR:
                env_proxy.action_space_str()

            elif op == EnvOperations.OBSERVATION_SPACE_STR:
                env_proxy.observation_space_str()

            # ---- Env lifecycle ----
            elif op == EnvOperations.RESET:
                env_proxy.reset()

            elif op == EnvOperations.STEP:
                action = socket.recv_json()
                env_proxy.step(action)

            elif op == EnvOperations.RENDER:
                env_proxy.render()

            elif op == EnvOperations.CLOSE:
                env_proxy.close()
                break

            else:
                socket.send_json({"error": "Unknown operation"})

    except Exception as ex:
        print(f"[!] Error: {ex}")

    finally:
        try:
            env_proxy.close()
        except Exception:
            pass

        socket.close()
        context.term()
        print("[+] Server terminated")
