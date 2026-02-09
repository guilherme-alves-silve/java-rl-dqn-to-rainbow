import zmq
import time
import gymnasium as gym

from enum import Enum


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

    def action_space_str(self):
        action_space_str = str(self.env.action_space)
        self.socket.send_json({ "actionSpaceStr": action_space_str })

    def observation_space_str(self):
        observation_space_str = str(self.env.observation_space)
        self.socket.send_json({ "observationSpaceStr": observation_space_str })

    def render(self):
        render = self.env.render()
        self._sent_once_render_metadata(render)
        self._send_and_wait(render)

    def action_space_sample(self):
        action = self.env.action_space.sample()
        self.socket.send_json({ "action": action })

    def reset(self):
        self.sent_render_metadata = False
        self.sent_state_metadata = False
        state, info = self.env.reset()
        self._sent_once_state_metadata(state)
        self.socket.send_json(info, zmq.SNDMORE)
        self._send_and_wait(state)

    def step(self, action):
        next_state, reward, term, trunc, info = self.env.step(action)
        self.socket.send_json({
            "reward": reward,
            "term": term,
            "trunc": trunc,
            "info": info
        }, zmq.SNDMORE)
        self._send_and_wait(next_state)

    def close(self):
        self.env.close()
        self.socket.send_json({ "close": True })

    def _sent_once_render_metadata(self, render):
        if not self.sent_render_metadata:
            metadata = {
                'shape': render.shape,
                'dtype': str(render.dtype)
            }
            print("SENDING RENDER METADATA")
            self.socket.send_json(metadata, zmq.SNDMORE)
            self.sent_render_metadata = True

    def _sent_once_state_metadata(self, state):
        if not self.sent_state_metadata:
            metadata = {
                'shape': state.shape,
                'dtype': str(state.dtype)
            }
            print("SENDING STATE METADATA")
            self.socket.send_json(metadata, zmq.SNDMORE)
            self.sent_state_metadata = True

    def _send_and_wait(self, data):
        tracker = self.socket.send(data, copy=False, track=True)
        tracker.wait()
