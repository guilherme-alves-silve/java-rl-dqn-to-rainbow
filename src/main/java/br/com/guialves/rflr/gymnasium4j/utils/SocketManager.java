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
    private ZMQ.Socket serverSocket;

    protected SocketManager(ZContext context, ZMQ.Socket serverSocket) {
        this.context = context;
        this.serverSocket = serverSocket;
    }

    public SocketManager(ZContext context) {
        this.context = context;
        this.serverSocket = newServerSocket();
    }

    public void reset() {
        this.serverSocket.close();
        this.serverSocket = newServerSocket();
    }

    private ZMQ.Socket newServerSocket() {
        var socket = context.createSocket(SocketType.REP);
        socket.setSendTimeOut(5000);
        socket.setReceiveTimeOut(5000);
        socket.setLinger(0);
        socket.bind("tcp://127.0.0.1:5555");
        log.info("Result from client: {}", socket.recvStr());
        return socket;
    }

    public String recvStr() {
        return serverSocket.recvStr();
    }

    public void sendStr(String data) {
        serverSocket.send(data);
    }

    public void recvByteBuffer(ByteBuffer buffer, int flags) {
        serverSocket.recvByteBuffer(buffer, flags);
    }

    public void sendJson(Object json) {
        serverSocket.send(GSON.toJson(json));
    }

    @Override
    public void close() {
        serverSocket.close();
    }
}
