package br.com.guialves.rflr.gymnasium4j;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.gymnasium4j.utils.EnvRenderWindow;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class App {

    static void main() {
        // TODO: Use VideoRecorder
        var envName = "CartPole-v1";
        var device = Device.gpu();
        try (var render = new EnvRenderWindow();
             var ndManager = NDManager.newBaseManager(device);
             var env = Gym.make(envName, ndManager)) {

            log.info("action space: {}, state space: {}", env.actionSpaceSampleDouble(), env.observationSpaceStr());

            while (!Thread.currentThread().isInterrupted()) {
                render.display(env.render());

                render.waitRender();
            }
        }
    }
}
