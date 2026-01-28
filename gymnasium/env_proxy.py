import zmq
import gymnasium as gym
from enum import Enum
from utils import create_socket


class EnvOperations(Enum):
    # Spaces
    ACTION_SPACE_SAMPLE = "1"
    ACTION_SPACE_STR = "2"
    OBSERVATION_SPACE_STR = "3"

    # Env lifecycle
    RESET = "4"
    STEP = "5"
    RENDER = "6"
    CLOSE = "7"


class EnvironmentProxy:

    def __init__(self, args, context, socket):
        self.args = args
        self.sent_render_metadata = False
        self.sent_state_metadata = False
        self.context = context
        self.socket = socket
        self.env = gym.make(args.env_name, render_mode="rgb_array", **args.env_params)

    def action_space_sample(self):
        action = self.env.action_space.sample()
        self.socket.send_json({ "action": action })

    def action_space_str(self):
        action_space_str = str(self.env.action_space)
        self.socket.send_json({ "action_space_str": action_space_str })

    def observation_space_str(self):
        observation_space_str = str(self.env.observation_space)
        self.socket.send_json({ "observation_space": observation_space_str })

    def render(self):
        render = self.env.render()
        if not self.sent_render_metadata:
            metadata = {
                'shape': render.shape,
                'dtype': str(render.dtype)
            }
            print("SENDING RENDER METADATA")
            self.socket.send_json(metadata, zmq.SNDMORE)
            self.sent_render_metadata = True
        tracker = self.socket.send(render, copy=False, track=True)
        tracker.wait()

    def reset(self):
        args = self.args
        self.sent_render_metadata = False
        self.sent_state_metadata = False
        self.context = zmq.Context()
        self.socket = create_socket(self.context, args.port, args.timeout)
        state, info = self.env.reset()
        if not self.sent_state_metadata:
            metadata = {
                'shape': state.shape,
                'dtype': str(state.dtype)
            }
            print("SENDING STATE METADATA")
            self.socket.send_json(metadata, zmq.SNDMORE)
            self.sent_state_metadata = True
        self.socket.send_json(info, zmq.SNDMORE)
        tracker = self.socket.send(state, copy=False, track=True)
        tracker.wait()

    def step(self, action):
        next_state, reward, term, trunc, info = self.env.step(action)
        self.socket.send_json({
            "reward": reward,
            "term": term,
            "trunc": trunc,
            "info": info
        }, zmq.SNDMORE)
        tracker = self.socket.send(next_state, copy=False, track=True)
        tracker.wait()

    def close(self):
        self.env.close()
        self.socket.send_json({ "close": True })
