package br.com.guialves.rflr.dqn.utils;

import org.zeromq.SocketType;
import org.zeromq.ZContext;
import org.zeromq.ZMQ;

import java.nio.ByteBuffer;

public class SocketManager {

    private final ZContext context;
    private ZMQ.Socket serverSocket;

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
        socket.bind("tcp://*:5555");
        return socket;
    }

    public String recvStr() {
        return serverSocket.recvStr();
    }

    public void send(String data) {
        serverSocket.send(data);
    }

    public void recvByteBuffer(ByteBuffer buffer, int flags) {
        serverSocket.recvByteBuffer(buffer, flags);
    }
}
