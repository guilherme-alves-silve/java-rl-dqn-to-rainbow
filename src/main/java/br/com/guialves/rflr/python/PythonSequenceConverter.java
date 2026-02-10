package br.com.guialves.rflr.python;

import java.util.ArrayList;

public class PythonSequenceConverter {

    static int[] parsePythonIntArray(String strShape) {
        if ("<NULL>".equalsIgnoreCase(strShape)) return null;

        var s = strShape.trim();

        if (s.startsWith("(")) s = s.substring(1);
        if (s.endsWith(")")) s = s.substring(0, s.length() - 1);

        s = s.trim();
        if (s.isEmpty()) {
            return new int[0];
        }

        String[] parts = s.split(",");
        var dims = new ArrayList<Integer>();

        for (String p : parts) {
            p = p.trim();
            if (p.isEmpty()) continue;
            dims.add(Integer.parseInt(p));
        }

        int[] shape = new int[dims.size()];
        for (int i = 0; i < dims.size(); i++) {
            shape[i] = dims.get(i);
        }

        return shape;
    }
}
