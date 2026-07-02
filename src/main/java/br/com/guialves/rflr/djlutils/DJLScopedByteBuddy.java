package br.com.guialves.rflr.djlutils;

import ai.djl.ndarray.NDManager;
import net.bytebuddy.ByteBuddy;
import net.bytebuddy.agent.ByteBuddyAgent;
import net.bytebuddy.asm.Advice;
import net.bytebuddy.dynamic.loading.ClassReloadingStrategy;
import net.bytebuddy.matcher.ElementMatchers;

public final class DJLScopedByteBuddy {

    private DJLScopedByteBuddy() {
        throw new IllegalStateException("No DJLScopedByteBuddy!");
    }

    public static void installOn(Class<?> type) {
        ByteBuddyAgent.install();
        new ByteBuddy()
                .redefine(type)
                .visit(Advice.to(NDManagerParameter0Advice.class)
                        .on(ElementMatchers.isAnnotatedWith(DJLScoped.class)
                                .and(ElementMatchers.takesArgument(0, NDManager.class))))
                .make()
                .load(type.getClassLoader(), ClassReloadingStrategy.fromInstalledAgent());
    }

    public static final class NDManagerParameter0Advice {

        @Advice.OnMethodEnter
        public static DJLMemoryManagement.Scope enter(@Advice.Argument(0) NDManager manager) {
            return DJLMemoryManagement.scoped(manager);
        }

        @Advice.OnMethodExit(onThrowable = Throwable.class)
        public static void exit(@Advice.Enter DJLMemoryManagement.Scope scope,
                                @Advice.Return(readOnly = false, typing = net.bytebuddy.implementation.bytecode.assign.Assigner.Typing.DYNAMIC) Object returned) {
            scope.close(returned);
        }
    }
}
