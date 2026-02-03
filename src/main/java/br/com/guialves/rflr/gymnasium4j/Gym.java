package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;
import br.com.guialves.rflr.gymnasium4j.utils.GymPythonLauncher;
import br.com.guialves.rflr.gymnasium4j.utils.SocketManager;
import lombok.SneakyThrows;
import org.zeromq.ZContext;

import java.util.Map;

public class Gym {

    public static EnvProxy make(String name, NDManager ndManager) {
        return make(name, "env_server.py", 5555, 5000, Map.of(), ndManager);
    }

    public static EnvProxy make(String name,
                                Map<String, Object> params,
                                NDManager ndManager) {
        return make(name, "env_server.py", 5555, 5000, params, ndManager);
    }

    @SneakyThrows
    public static EnvProxy make(String name,
                                String scriptPath,
                                int port,
                                int timeout,
                                Map<String, Object> params,
                                NDManager ndManager) {
        var launcher = new GymPythonLauncher(scriptPath, port, timeout, name, params);
        var context = new ZContext();
        var socket = new SocketManager(context);
        return new EnvProxy(context, socket, launcher, ndManager);
    }
}
