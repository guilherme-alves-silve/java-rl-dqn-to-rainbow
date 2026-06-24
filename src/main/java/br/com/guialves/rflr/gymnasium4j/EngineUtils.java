package br.com.guialves.rflr.gymnasium4j;

import ai.djl.engine.Engine;
import ai.djl.training.GradientCollector;

public class EngineUtils {

    private EngineUtils() {
        throw new IllegalArgumentException("No EngineUtils!");
    }

    public static GradientCollector gradCol() {
        return Engine.getInstance().newGradientCollector();
    }
}
