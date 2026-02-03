package br.com.guialves.rflr.gymnasium4j.transform;

import br.com.guialves.rflr.gymnasium4j.utils.SocketManager;

/**
 * Wire-protocol operations for the Gymnasium ZeroMQ server.
 *
 * MUST stay in sync with:
 *   env_proxy.py → class EnvOperations(Enum)
 */
public enum EnvOperations {

    // -------- Spaces --------
    ACTION_SPACE_SAMPLE("1"),
    ACTION_SPACE_STR("2"),
    OBSERVATION_SPACE_STR("3"),

    // -------- Environment lifecycle --------
    RESET("4"),
    STEP("5"),
    RENDER("6"),
    CLOSE("7") {
        @Override
        public void exec(SocketManager socket) {
            socket.sendStr(value);
            socket.close();
        }
    };

    protected final String value;

    EnvOperations(String value) {
        this.value = value;
    }

    /**
     * Value sent over the wire (ZMQ frame).
     */
    public String value() {
        return value;
    }

    /**
     * Convert a wire value back into an enum.
     * Useful for debugging or ROUTER/DEALER setups.
     */
    public static EnvOperations fromValue(String value) {
        for (EnvOperations op : values()) {
            if (op.value.equals(value)) {
                return op;
            }
        }
        throw new IllegalArgumentException("Unknown EnvOperation: " + value);
    }

    public void exec(SocketManager socket) {
        socket.sendStr(value);
    }
}
