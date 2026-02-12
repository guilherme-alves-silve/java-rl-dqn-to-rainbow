package br.com.guialves.rflr.python;

import org.jspecify.annotations.NonNull;

import java.util.HashMap;
import java.util.Map;

class PythonDictConverter {

    private PythonDictConverter() {
        throw new IllegalArgumentException("No PythonDictConverter!");
    }

    static Map<Object, Object> parsePythonDictRepr(String repr) {
        var map = new HashMap<>();

        // remove { }
        String body = repr.trim();
        if (body.startsWith("{")) body = body.substring(1);
        if (body.endsWith("}")) body = body.substring(0, body.length() - 1);

        if (body.isBlank()) {
            return map;
        }

        var entries = body.split(", (?=(?:[^']*'[^']*')*[^']*$)");

        for (String entry : entries) {
            String[] kv = entry.split(":", 2);
            if (kv.length != 2) continue;

            String keyStr = kv[0].trim();
            String valueStr = kv[1].trim();

            // Parse key (supports strings, numbers, None, True, False)
            Object key = parsePythonKey(keyStr);
            if (key == null) continue; // Skip if key parsing fails

            map.put(key, parsePythonValue(valueStr));
        }

        return map;
    }

    private static Object parsePythonKey(String key) {
        // Check for None
        if ("None".equals(key)) {
            return null;
        }

        // Check for booleans
        if ("True".equals(key)) {
            return Boolean.TRUE;
        }
        if ("False".equals(key)) {
            return Boolean.FALSE;
        }

        // Check for string keys (quoted)
        return getObject(key);
    }

    private static Object parsePythonValue(String value) {
        switch (value) {
            case "None" -> {
                return null;
            }
            case "True" -> {
                return Boolean.TRUE;
            }
            case "False" -> {
                return Boolean.FALSE;
            }
        }

        return getObject(value);
    }

    @NonNull
    private static Object getObject(String value) {
        if ((value.startsWith("'") && value.endsWith("'")) ||
                (value.startsWith("\"") && value.endsWith("\""))) {
            return value.substring(1, value.length() - 1);
        }

        try {
            if (value.contains(".")) return Double.parseDouble(value);
            return Long.parseLong(value);
        } catch (NumberFormatException ignored) {
        }

        return value;
    }
}
