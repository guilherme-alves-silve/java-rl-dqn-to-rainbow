package br.com.guialves.rflr.execs;

/**
 * CLI-driven run options. Null fields fall back to the default
 * behavior (i.e. train + save + run once).
 */
public record RLRunOptions(String loadModelPrefix) {

    public static RLRunOptions defaults() {
        return new RLRunOptions(null);
    }
}
