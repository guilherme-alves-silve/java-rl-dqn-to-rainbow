package br.com.guialves.rflr.utils;

public class PropUtils {

    private PropUtils() {
        throw new IllegalStateException("No PropUtils!");
    }

    public static int getIntProp(String key, String value) {
        return Integer.parseInt(System.getProperty(key, value));
    }

    public static boolean getBoolProp(String key, String value) {
        return Boolean.parseBoolean(System.getProperty(key, value));
    }

    public static boolean getBoolProp(String key) {
        return getBoolProp(key, "true");
    }
}
