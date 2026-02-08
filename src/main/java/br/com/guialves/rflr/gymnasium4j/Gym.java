package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.gymnasium4j.utils.GymPythonLauncher;
import br.com.guialves.rflr.gymnasium4j.utils.SocketManager;
import io.vavr.control.Try;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.zeromq.ZContext;

import java.util.Map;

@Slf4j
public class Gym {

    public static EnvProxy make(String name, NDManager ndManager) {
        return make(name, "env_server.py", 5555, 30_000, false, Map.of(), ndManager);
    }

    public static EnvProxy make(String name,
                                Map<String, Object> params,
                                NDManager ndManager) {
        return make(name, "env_server.py", 5555, 30_000, false, params, ndManager);
    }

    @SneakyThrows
    public static EnvProxy make(String name,
                                String scriptPath,
                                int port,
                                int timeout,
                                boolean debug,
                                Map<String, Object> params,
                                NDManager ndManager) {
        var launcher = new GymPythonLauncher(scriptPath, port, timeout, debug, name, params);
        launcher.start();
        var context = new ZContext();
        var socket = new SocketManager(context);
        var env = new EnvProxy(context, socket, launcher, ndManager);
        Runtime.getRuntime().addShutdownHook(new Thread(env::close));
        return env;
    }
}
