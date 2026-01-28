package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.gymnasium4j.dto.EnvStatus;
import br.com.guialves.rflr.gymnasium4j.transform.GymReader;
import br.com.guialves.rflr.gymnasium4j.utils.EnvRenderWindow;
import br.com.guialves.rflr.gymnasium4j.utils.SocketManager;
import lombok.SneakyThrows;
import org.zeromq.ZContext;

public class App {
    @SneakyThrows
    static void main() {

        var gymReader = new GymReader();
        // TODO: Use VideoRecorder
        try (var context = new ZContext();
             var render = new EnvRenderWindow()) {
            var socket = new SocketManager(context);

            while (!Thread.currentThread().isInterrupted()) {

                var image = gymReader.getImage(socket);
                render.displayImage(image);
                Thread.sleep(1000/60);
                EnvStatus.sampleAction(socket);
            }
        }

        try (var manager = NDManager.newBaseManager()) {
            var data = manager.create(new float[][] {{1, 2}, {3, 4}, {5, 6}});
            var indices = manager.create(new long[] {0, 2});
            var result = data.gather(indices, 0); // Returns [[1, 2], [5, 6]]
        }
    }
}
