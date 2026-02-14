package br.com.guialves.rflr.python.numpy;

import lombok.Getter;
import org.bytedeco.cpython.PyObject;
import org.bytedeco.cpython.Py_buffer;

import java.nio.ByteBuffer;

import static org.bytedeco.cpython.global.python.PyBUF_SIMPLE;
import static org.bytedeco.cpython.global.python.PyBuffer_Release;
import static org.bytedeco.cpython.global.python.PyObject_GetBuffer;

public class NumPyBufferView implements AutoCloseable {
    private final Py_buffer view;
    @Getter
    private final ByteBuffer buffer;

    public NumPyBufferView(PyObject ndarray) {
        this.view = new Py_buffer();
        int rc = PyObject_GetBuffer(ndarray, view, PyBUF_SIMPLE);
        if (rc != 0) {
            throw new IllegalStateException("PyObject_GetBuffer failed");
        }

        long size = view.len();
        this.buffer = view.buf().capacity(size).asByteBuffer();
    }

    @Override
    public void close() {
        PyBuffer_Release(view);
    }
}
