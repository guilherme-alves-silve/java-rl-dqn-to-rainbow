package br.com.guialves.rflr.python;

import org.bytedeco.cpython.PyObject;

import java.util.HashMap;
import java.util.Map;

class PythonDictConverter {

    private PythonDictConverter() {
        throw new IllegalArgumentException("No PythonDictConverter!");
    }

    static Map<String, Object> parsePythonDictRepr(String repr) {
        var map = new HashMap<String, Object>();

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

            String key = kv[0].trim();
            String value = kv[1].trim();

            // key: 'abc'
            key = key.replaceAll("^['\"]|['\"]$", "");

            map.put(key, parsePythonValue(value));
        }

        return map;
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
