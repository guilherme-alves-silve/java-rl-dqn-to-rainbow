package br.com.guialves.rflr.gymnasium4j.wrappers;

import java.util.ArrayList;

/**
 * Bridge to the wrappers.RecordEpisodeStatistics from Python gymnasium.
 * Reference:
 * <a href="https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordEpisodeStatistics">RecordEpisodeStatistics</a>
 */
public record RecordEpisodeStatistics(int bufferLength, String statsKey) implements IWrapper {

    public RecordEpisodeStatistics() {
        this(-1, null);
    }

    @Override
    public String pyToStr(String varName) {
        var args = new ArrayList<String>();
        args.add(varName);
        if (bufferLength != -1) args.add("buffer_length=%d".formatted(bufferLength));
        if (statsKey != null && !statsKey.isBlank()) args.add("stats_key='%s'".formatted(statsKey));
        return "%s(%s)".formatted(this.getClass().getSimpleName(), String.join(", ", args));
    }
}
