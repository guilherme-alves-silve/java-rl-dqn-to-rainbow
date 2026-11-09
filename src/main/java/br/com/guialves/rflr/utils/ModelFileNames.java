package br.com.guialves.rflr.utils;

import lombok.SneakyThrows;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Pattern;

/**
 * Naming rules for the files that persist trained models.
 * <p>
 * The environment name is not always a plain file name: Atari environments are
 * registered as {@code ALE/Galaxian-v5}. DJL resolves the parameters file as
 * {@code modelPath.resolve(prefix + "-0000.params")}, so a separator inside the
 * prefix silently redirects to write to a nested folder
 * ({@code output_models/<algorithm>/ALE/...}) that nobody creates, and the save
 * fails with {@link java.nio.file.NoSuchFileException}. DJL only creates the base
 * folder ({@code modelPath}) before writing.
 *
 * @see br.com.guialves.rflr.execs.RLConfig#modelFileName()
 */
public final class ModelFileNames {

    private static final Pattern ILLEGAL_CHARS = Pattern.compile("[\\\\/:*?\"<>|]+");
    private static final Pattern CONTROL_CHARS = Pattern.compile("[\\p{Cntrl}\\s]+");
    private static final Pattern REPEATED_DASHES = Pattern.compile("-{2,}");
    private static final Pattern EDGE_DASHES = Pattern.compile("^-+|-+$");

    private ModelFileNames() {
        throw new IllegalStateException("No ModelFileNames!");
    }

    /**
     * Replaces every character that cannot be part of a file name, so the result can
     * safely be used as a DJL model prefix.
     *
     * @param name raw name, usually the environment name
     * @return the name without path separators nor reserved characters
     */
    public static String sanitize(String name) {
        if (name == null || name.isBlank()) {
            return "model";
        }
        var sanitized = ILLEGAL_CHARS.matcher(name).replaceAll("-");
        sanitized = CONTROL_CHARS.matcher(sanitized).replaceAll("-");
        sanitized = REPEATED_DASHES.matcher(sanitized).replaceAll("-");
        return EDGE_DASHES.matcher(sanitized).replaceAll("");
    }

    /**
     * Builds the name of the file that holds the model trained for the given
     * environment, without the {@code -NNNN.params} suffix added by DJL.
     *
     * @param envName      environment name, e.g. {@code ALE/Galaxian-v5}
     * @param algorithmName algorithm name, e.g. {@code galaxian_noisy_nets_dqn}
     * @return a file name safe to be used as a DJL model prefix
     */
    public static String build(String envName, String algorithmName) {
        return "%s_%d_%s".formatted(sanitize(envName),
                System.currentTimeMillis(),
                sanitize(algorithmName));
    }

    /**
     * Creates the folder that will hold the parameters file, so DJL never fails
     * with {@link java.nio.file.NoSuchFileException} while writing it.
     *
     * @param modelPath folder the model is saved into
     * @param prefix    model file name, without the epoch suffix added by DJL
     */
    @SneakyThrows
    public static void ensureParentDirectory(Path modelPath, String prefix) {
        var parent = modelPath.resolve(prefix).getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
    }
}
