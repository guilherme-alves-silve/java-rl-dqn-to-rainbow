import zmq
import argparse


def create_socket(context, port: int, timeout: int):
    socket = context.socket(zmq.REP)
    socket.connect(f"tcp://127.0.0.1:{port}")
    socket.setsockopt(zmq.RCVTIMEO, timeout)
    socket.setsockopt(zmq.SNDTIMEO, timeout)
    # Used to avoid waiting forever.
    socket.setsockopt(zmq.LINGER, 0)
    return socket


def str2bool(value):
    if isinstance(value, bool):
        return value

    value = value.strip().lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off", ""):
        return False

    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: '{value}'. Use true/false."
    )
