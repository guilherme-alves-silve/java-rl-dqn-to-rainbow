from env_proxy import EnvOperations


def test_env_operations_enum_values():
    assert EnvOperations.ACTION_SPACE_SAMPLE.value == "1"
    assert EnvOperations.RESET.value == "4"
    assert EnvOperations.CLOSE.value == "7"


def test_dispatch_action_space_sample(mocker):
    socket = mocker.Mock()
    socket.recv_string.return_value = EnvOperations.ACTION_SPACE_SAMPLE.value

    env_proxy = mocker.Mock()

    cmd = socket.recv_string()
    op = EnvOperations(cmd)

    if op == EnvOperations.ACTION_SPACE_SAMPLE:
        env_proxy.action_space_sample()

    env_proxy.action_space_sample.assert_called_once()


def test_dispatch_step_reads_action(mocker):
    socket = mocker.Mock()
    socket.recv_string.return_value = EnvOperations.STEP.value
    socket.recv_json.return_value = 1

    env_proxy = mocker.Mock()

    cmd = socket.recv_string()
    op = EnvOperations(cmd)

    if op == EnvOperations.STEP:
        action = socket.recv_json()
        env_proxy.step(action)

    env_proxy.step.assert_called_once_with(1)
