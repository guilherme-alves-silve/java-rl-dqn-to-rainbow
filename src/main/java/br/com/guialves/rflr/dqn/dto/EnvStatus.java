package br.com.guialves.rflr.dqn.dto;

import br.com.guialves.rflr.dqn.utils.SocketManager;
import lombok.NoArgsConstructor;
import org.zeromq.ZMQ;

@NoArgsConstructor
public class EnvStatus {

    public static final String REPORT = "1";
    public static final String RENDER = "2";
    public static final String METADATA_RENDER = "3";
    public static final String SAMPLE_ACTION = "4";
    public static final String DISCRETE_ACTION = "5";
    public static final String CONTINUOUS_ACTION = "6";
    public static final String DONE = "7";

    public static boolean mustReport(String result) {
        return REPORT.equals(result);
    }

    public static void sendImgToRender(SocketManager socket) {
        socket.send(EnvStatus.RENDER);
    }

    public static void sendMetadataOfRender(SocketManager socket) {
        socket.send(EnvStatus.METADATA_RENDER);
    }

    public static void sampleAction(SocketManager socket) {
        socket.send(SAMPLE_ACTION);
    }
}
