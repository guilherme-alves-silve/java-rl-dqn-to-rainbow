import zmq
from utils import create_socket


def test_create_socket_sets_options(mocker):
    context = mocker.Mock()
    socket = mocker.Mock()
    context.socket.return_value = socket

    result = create_socket(context, 5555, 1000)

    context.socket.assert_called_once_with(zmq.REP)
    socket.connect.assert_called_once_with("tcp://127.0.0.1:5555")

    socket.setsockopt.assert_any_call(zmq.RCVTIMEO, 1000)
    socket.setsockopt.assert_any_call(zmq.SNDTIMEO, 1000)
    socket.setsockopt.assert_any_call(zmq.LINGER, 0)

    assert result == socket
