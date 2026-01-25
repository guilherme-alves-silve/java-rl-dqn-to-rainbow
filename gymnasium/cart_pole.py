# https://gymnasium.farama.org/environments/classic_control/cart_pole/

import zmq
import gymnasium as gym

REPORT = "1"
RENDER = "2"
METADATA_RENDER = "3"
SAMPLE_ACTION = "4"
DISCRETE_ACTION = "5"
CONTINUOUS_ACTION = "6"
DONE = "7"

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

env_name = "CartPole-v1"
env = gym.make(env_name,
               render_mode="rgb_array")
env.reset()

socket.setsockopt(zmq.RCVTIMEO, 5000)
socket.setsockopt(zmq.SNDTIMEO, 5000)
socket.setsockopt(zmq.LINGER, 0)

try:
    sent_metadata = False
    for step in range(500):

        render = None
        if not sent_metadata:
            render = env.render()
            metadata = {
                'shape': render.shape,
                'dtype': str(render.dtype)
            }
            print("SENDING METADATA")
            socket.send_json(metadata, zmq.SNDMORE)
            sent_metadata = True
        print(f"SENDING RENDER: {step}")
        render = env.render()
        socket.send(render)

        resp = socket.recv_string()
        print(f"resp: {resp}")
        if SAMPLE_ACTION == resp:
            print("SAMPLE_ACTION")
            action = env.action_space.sample()
        elif DISCRETE_ACTION == resp:
            print("DISCRETE_ACTION")
            action = 0
        elif CONTINUOUS_ACTION == resp:
            print("CONTINUOUS_ACTION")
            action = 0

        state, reward, term, trunc, info = env.step(action)

        # Who decides if it's done is the java side
        if term or trunc:
            print("SENDING DONE")
            socket.send_string(DONE)
            env.reset()
            break
except Exception as ex:
    print(f"Error: {ex}")
finally:
    env.close()
    socket.close()
    context.term()
