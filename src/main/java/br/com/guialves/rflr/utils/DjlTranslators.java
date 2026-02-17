package br.com.guialves.rflr.utils;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

public class DjlTranslators {

    public static final SimpleDjlTranslator SIMPLE_DJL_TRANSLATOR = new SimpleDjlTranslator();

    public static class SimpleDjlTranslator implements Translator<NDArray, NDArray> {

        @Override
        public NDList processInput(TranslatorContext ctx, NDArray input) {
            return new NDList(input);
        }

        @Override
        public NDArray processOutput(TranslatorContext ctx, NDList list) {
            return list.singletonOrThrow();
        }
    }
}
