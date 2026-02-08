package br.com.guialves.rflr.gymnasium4j.utils;

import ai.djl.util.JsonUtils;
import com.google.gson.Gson;
import lombok.extern.slf4j.Slf4j;
import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

import java.nio.ByteBuffer;

@Slf4j
public class SocketManager implements AutoCloseable {

    private static final Gson GSON = JsonUtils.GSON_COMPACT;
    private final ZContext context;
    private final ZMQ.Socket socket;

    SocketManager(ZContext context, ZMQ.Socket socket) {
        this.context = context;
        this.socket = socket;
    }

    public SocketManager(ZContext context) {
        this.context = context;
        this.socket = newServerSocket();
    }

    private ZMQ.Socket newServerSocket() {
        var socket = context.createSocket(SocketType.REQ);
        socket.setSendTimeOut(Integer.parseInt(System.getProperty("zmq.send.timeout", "5000")));
        socket.setReceiveTimeOut(Integer.parseInt(System.getProperty("zmq.receive.timeout", "5000")));
        socket.setLinger(0);
        socket.bind("tcp://127.0.0.1:5555");
        return socket;
    }

    public String recvStr() {
        return socket.recvStr();
    }

    public void sendStr(String data) {
        socket.send(data);
    }

    public void recvByteBuffer(ByteBuffer buffer, int flags) {
        socket.recvByteBuffer(buffer, flags);
    }

    public void sendJson(Object json) {
        socket.send(GSON.toJson(json));
    }

    @Override
    public void close() {
        socket.close();
    }
}
