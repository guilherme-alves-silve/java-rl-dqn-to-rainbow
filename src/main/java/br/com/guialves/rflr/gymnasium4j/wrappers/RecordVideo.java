package br.com.guialves.rflr.gymnasium4j.wrappers;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;

/**
 * Bridge to the wrappers.RecordVideo from Python gymnasium.
 * Reference:
 * <a href="https://gymnasium.farama.org/api/wrappers/misc_wrappers/#gymnasium.wrappers.RecordVideo">RecordVideo</a>
 */
public record RecordVideo(String videoFolder,
                          int episodeTrigger,
                          int stepTrigger,
                          int videoLength,
                          String namePrefix,
                          int fps,
                          boolean disableLogger,
                          int gcTrigger) implements IWrapper {

    public RecordVideo(Path videoFolder) {
        this(videoFolder.toString());
    }

    public RecordVideo(String videoFolder) {
        this(videoFolder, -1, -1, -1, null, -1, false, -1);
    }

    @Override
    public String pyToStr(String varName) {
        var args = new ArrayList<String>();
        args.add(varName);
        args.add("video_folder='%s'".formatted(videoFolder.replace("\\", "/")));
        addParam("episode_trigger", args, episodeTrigger);
        addParam("step_trigger", args, stepTrigger);
        args.add("video_length=%s".formatted(videoLength == -1? 0 : videoLength));
        if (namePrefix != null && !namePrefix.isBlank()) args.add("name_prefix='%s'".formatted(namePrefix));
        else args.add("name_prefix='rl-video'");
        if (fps != -1) args.add("fps=%d".formatted(fps));
        if (disableLogger) args.add("disable_logger=True");
        if (gcTrigger != -1) args.add("gc_trigger=%d".formatted(gcTrigger));
        return "%s(%s)".formatted(this.getClass().getSimpleName(), String.join(", ", args));
    }

    private void addParam(String paramName,
                          ArrayList<String> args,
                          int value) {
        if (value != -1) args.add("%s=lambda t: t %% %d == 0".formatted(paramName, value));
        else args.add("%s=None".formatted(paramName));
    }
}
