package br.com.guialves.rflr.gymnasium4j;

import ai.djl.ndarray.NDManager;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Gym {

    @SneakyThrows
    public static Env make(String name,
                           NDManager ndManager) {
        var env = new Env(name, ndManager);
        Runtime.getRuntime().addShutdownHook(new Thread(env::close));
        return env;
    }
}
