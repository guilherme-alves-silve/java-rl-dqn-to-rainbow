import zmq
import pytest
import numpy as np

from env_proxy import EnvironmentProxy


class DummyArgs:
    env_name = "CartPole-v1"
    env_params = {}
    port = 5555
    timeout = 1000


@pytest.fixture
def mock_env(mocker):
    env = mocker.Mock()

    env.action_space.sample.return_value = 1
    env.action_space.__str__ = lambda _: "Discrete(2)"
    env.observation_space.__str__ = lambda _: "Box(4,)"

    env.reset.return_value = (
        np.zeros((4,), dtype=np.float32),
        {"reset": True},
    )

    env.step.return_value = (
        np.ones((4,), dtype=np.float32),
        1.0,
        False,
        False,
        {"step": True},
    )

    env.render.return_value = np.zeros((64, 64, 3), dtype=np.uint8)

    return env


@pytest.fixture
def mock_socket(mocker):
    socket = mocker.Mock()

    tracker = mocker.Mock()
    tracker.wait.return_value = None

    socket.send.return_value = tracker
    socket.send_json.return_value = None

    return socket


@pytest.fixture
def env_proxy(mocker, mock_env, mock_socket):
    mocker.patch("gymnasium.make", return_value=mock_env)
    context = mocker.Mock()
    args = DummyArgs()

    return EnvironmentProxy(args, context, mock_socket)


def test_action_space_sample(env_proxy, mock_socket):
    env_proxy.action_space_sample()

    mock_socket.send_json.assert_called_once()
    payload = mock_socket.send_json.call_args[0][0]
    assert "action" in payload


def test_action_space_str(env_proxy, mock_socket):
    env_proxy.action_space_str()

    payload = mock_socket.send_json.call_args[0][0]
    assert payload["actionSpaceStr"] == "Discrete(2)"


def test_observation_space_str(env_proxy, mock_socket):
    env_proxy.observation_space_str()

    payload = mock_socket.send_json.call_args[0][0]
    assert payload["observationSpaceStr"] == "Box(4,)"


def test_render_sends_metadata_once(env_proxy, mock_socket):
    env_proxy.render()
    env_proxy.render()

    # metadata sent only once
    send_json_calls = mock_socket.send_json.call_args_list
    assert len(send_json_calls) == 1

    metadata, flags = send_json_calls[0][0]
    assert "shape" in metadata
    assert "dtype" in metadata
    assert flags == zmq.SNDMORE

    assert mock_socket.send.call_count == 2


def test_reset_recreates_socket_and_sends_state(env_proxy, mocker):

    env_proxy.reset()

    assert env_proxy.sent_render_metadata is False
    assert env_proxy.sent_state_metadata is True
    assert env_proxy.socket.send_json.call_count >= 2
    assert env_proxy.socket.send.called


def test_step_sends_reward_and_state(env_proxy, mock_socket):
    env_proxy.step(1)

    payload, flags = mock_socket.send_json.call_args[0]
    assert payload["reward"] == 1.0
    assert flags == zmq.SNDMORE

    mock_socket.send.assert_called_once()


def test_close_sends_close_flag(env_proxy, mock_socket):
    env_proxy.close()

    mock_socket.send_json.assert_called_once_with({"close": True})
