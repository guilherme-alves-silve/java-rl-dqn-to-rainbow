import zmq


def create_socket(context, port: int, timeout: int):
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://127.0.0.1:{port}")
    socket.setsockopt(zmq.RCVTIMEO, timeout)
    socket.setsockopt(zmq.SNDTIMEO, timeout)
    # Used to avoid waiting forever.
    socket.setsockopt(zmq.LINGER, 0)
    return socket
