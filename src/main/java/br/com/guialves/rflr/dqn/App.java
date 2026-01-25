package br.com.guialves.rflr.dqn;

import ai.djl.modality.cv.ImageFactory;
import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.dqn.transform.GymReader;
import br.com.guialves.rflr.dqn.utils.EnvRenderWindow;
import org.zeromq.SocketType;
import org.zeromq.ZContext;

public class App {
    static void main() {

        var gymReader = new GymReader();
        // TODO: Use VideoRecorder
        try (var context = new ZContext();
             var render = new EnvRenderWindow()) {
            var socket = context.createSocket(SocketType.REP);
            socket.bind("tcp://*:5555");

            while (!Thread.currentThread().isInterrupted()) {

                var image = gymReader.getImage(socket);
                render.displayImage(image);
                socket.send("ACK");
            }
        }

        //try (var manager = NDManager.newBaseManager()) {
        //    manager.create()
        //}
    }
}
