package br.com.guialves.rflr.djlutils;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Marks a method whose temporary DJL arrays should be released after execution.
 *
 * <p>The annotation is activated by Byte Buddy instrumentation. The injected
 * bytecode opens a {@link DJLMemoryManagement.Scope} before the method body and
 * closes it after the method exits, preserving a returned NDArray/NDList.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
public @interface DJLScoped {

    /**
     * Argument index containing the NDManager to scope. The current zero-reflection
     * Byte Buddy advice supports index 0; add a dedicated advice for other hot
     * paths instead of using reflection.
     */
    int managerParameter() default 0;
}
