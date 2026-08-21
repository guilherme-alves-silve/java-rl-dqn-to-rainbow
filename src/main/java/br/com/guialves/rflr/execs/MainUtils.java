package br.com.guialves.rflr.execs;

import java.util.regex.Pattern;

public class MainUtils {

    private static final Pattern MODEL_NAME_PATTERN = Pattern.compile(".+-(\\d{4}).params");

    private MainUtils() {
        throw new IllegalStateException("No MainUtils!");
    }

    static RLRunOptions parseArgs(String[] args, String className) {
        String loadPrefix = null;
        for (int i = 0; i < args.length; i++) {
            var arg = args[i];
            switch (arg) {
                case "--load-model", "-l" -> {
                    if (i + 1 >= args.length) {
                        throw new IllegalArgumentException(
                                "--load-model requires a prefix argument");
                    }
                    loadPrefix = args[++i];
                    final var matcher = MODEL_NAME_PATTERN.asMatchPredicate();
                    if (matcher.test(loadPrefix)) {
                        throw new IllegalArgumentException("Pass only the name of the model, remove the \"-NUMBERS.params\"");
                    }
                }
                case "--help", "-h" -> {
                    printUsage(className);
                    System.exit(0);
                }
                default -> throw new IllegalArgumentException("Unknown argument: " + arg);
            }
        }
        return new RLRunOptions(loadPrefix);
    }

    private static void printUsage(String className) {
        IO.println("""
            Usage: %s [options]

            Options:
              --load-model <prefix>, -l <prefix>   Load a trained model whose file name
                                                  starts with <prefix> from
                                                  ./output_models/<algorithm>/ and only
                                                  run evaluation (no training, no saving).
              --help, -h                           Show this help message.

            Without options the agent trains a new model, saves it, and runs one
            evaluation episode.

            Example:
              java %s --load-model LunarLander-v3_1787202869783_dqn
            """.formatted(className, className));
    }
}
