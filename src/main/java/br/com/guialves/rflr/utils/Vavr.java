package br.com.guialves.rflr.utils;

import io.vavr.CheckedFunction0;
import io.vavr.control.Try;

public class Vavr {

    private Vavr() {
        throw new IllegalArgumentException("No Vavr!");
    }

    public static <T> Try<T> retry(int times, CheckedFunction0<T> supplier) {
        return Try.of(supplier)
                .recoverWith(ex -> times > 1 ? retry(times - 1, supplier) : Try.failure(ex));
    }
}
