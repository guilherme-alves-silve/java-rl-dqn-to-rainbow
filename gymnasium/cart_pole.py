# https://gymnasium.farama.org/environments/classic_control/cart_pole/

import zmq
import gymnasium as gym

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

env_name = "CartPole-v1"
env = gym.make(env_name,
               render_mode="rgb_array")
env.reset()

sent_metadata = False
for step in range(500):
    render = env.render()

    if not sent_metadata:
        metadata = {
            'shape': render.shape,
            'dtype': str(render.dtype),
            'order': 'C'
        }
        socket.send_json(metadata, zmq.SNDMORE)
        sent_metadata = True
    socket.send(render)

    response = socket.recv()
    print(f"Step {step}: {response.decode()}")

    action = env.action_space.sample()
    a = env.step(action)
    print(a)
    state, reward, term, trunc, info = a

    if term or trunc:
        env.reset()
        break

env.close()
socket.close()
context.term()