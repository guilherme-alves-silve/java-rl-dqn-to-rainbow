package br.com.guialves.rflr.python.numpy;

import lombok.NonNull;
import lombok.experimental.Delegate;
import org.bytedeco.cpython.PyObject;
import org.bytedeco.cpython.Py_buffer;

import java.nio.ByteBuffer;

import static org.bytedeco.cpython.global.python.*;

/**
 * Class used to work with zero-copy between Python and Java.
 * <p>References:
 * <ul>
 *   <li>CPython C API – Buffer Protocol:
 *       <a href="https://docs.python.org/3/c-api/buffer.html">...</a></li>
 *   <li>PEP 3118 – Revising the Buffer Protocol:
 *       <a href="https://peps.python.org/pep-3118/">...</a></li>
 * </ul>
 * </p>
 */
public class NumPyBufferView implements AutoCloseable {
    private final Py_buffer view;
    @Delegate
    private final ByteBuffer buffer;

    public NumPyBufferView(@NonNull PyObject ndarray) {
        this.view = new Py_buffer();
        int rc = PyObject_GetBuffer(ndarray, view, PyBUF_SIMPLE);
        if (rc != 0) {
            throw new IllegalStateException("PyObject_GetBuffer failed (array not contiguous?), return code: " + rc);
        }

        long size = view.len();
        this.buffer = view.buf().capacity(size).asByteBuffer();
    }

    /**
     * Used internally!
     */
    ByteBuffer buffer() {
        return buffer;
    }

    @Override
    public void close() {
        PyBuffer_Release(view);
    }
}
