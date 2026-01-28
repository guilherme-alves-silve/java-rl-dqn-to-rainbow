package br.com.guialves.rflr.gymnasium4j.utils;

import lombok.SneakyThrows;
import org.bytedeco.ffmpeg.global.avcodec;
import org.bytedeco.ffmpeg.global.avutil;
import org.bytedeco.javacv.FFmpegFrameRecorder;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.Frame;

import java.awt.image.BufferedImage;

public class VideoRecorder implements AutoCloseable {

    private final FFmpegFrameRecorder recorder;
    private final Java2DFrameConverter converter;

    @SneakyThrows
    public VideoRecorder(String outputPath, int width, int height, int fps) {
        recorder = new FFmpegFrameRecorder(outputPath, width, height);
        recorder.setVideoCodec(avcodec.AV_CODEC_ID_H264);
        recorder.setFormat("mp4");
        recorder.setFrameRate(fps);
        recorder.setPixelFormat(avutil.AV_PIX_FMT_YUV420P);

        converter = new Java2DFrameConverter();
        recorder.start();
    }

    private void start() throws Exception {
        // TODO: Check thread safety problems later
        recorder.start();
    }

    public void addFrame(BufferedImage image) throws Exception {
        Frame frame = converter.convert(image);
        recorder.record(frame);
    }

    private void finish() throws Exception {
        recorder.stop();
        recorder.release();
        converter.close();
    }

    @Override
    public void close() throws Exception {
        finish();
    }
}
