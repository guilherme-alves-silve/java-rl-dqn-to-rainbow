package br.com.guialves.rflr.python;

import org.bytedeco.cpython.PyObject;
import org.bytedeco.javacpp.Pointer;

import static org.bytedeco.cpython.global.python.*;
import static org.bytedeco.cpython.global.python.PyGILState_Release;

public final class PyRef extends PyObject implements AutoCloseable {

    private final boolean useGil;

    /**
     * Copies the address of the owned reference. The copy constructor of
     * {@link Pointer} acts as a {@code reinterpret_cast}; since {@link PyObject} has
     * no deallocator, no {@code ProxyDeallocator} is created and nothing else is
     * affected.
     */
    private PyRef(boolean useGil, Pointer owned) {
        super(owned);
        this.useGil = useGil;
    }

    /**
     * Takes ownership of a Python reference, so that it gets released on
     * {@link #close()}.
     *
     * @param owned a reference whose ownership was transferred to the caller, as
     *              returned by {@code PyObject_GetAttrString()} and the like. May
     *              be {@code null} or a null pointer, in which case the returned
     *              object does nothing when closed.
     * @return a {@code PyRef} that releases {@code owned} on {@link #close()}
     */
    public static PyRef of(PyObject owned) {
        return new PyRef(false, owned);
    }

    public static PyRef ofGIL(PyObject owned) {
        return new PyRef(true, owned);
    }

    /**
     * Returns the Python object this instance refers to, that is, this instance
     * itself.
     *
     * @return the referenced Python object
     */
    public PyObject get() {
        return this;
    }

    /**
     * Returns the current reference count of the Python object, as reported by
     * {@code Py_REFCNT()}.
     *
     * @return the reference count, or {@code -1} if the pointer is null
     */
    public long refCount() {
        return isNull() ? -1 : Py_REFCNT(this);
    }

    /**
     * Releases the owned reference by calling {@code Py_DECREF()} exactly once, and
     * invalidates the pointer. Calling this method more than once is safe: the
     * subsequent calls do nothing.
     */
    @Override
    public void close() {
        if (isNull()) {
            return;
        }

        var self = this;
        if (useGil) {
            int state = PyGILState_Ensure();
            try {
                Py_DECREF(self);
            } finally {
                PyGILState_Release(state);
            }
        } else {
            Py_DECREF(self);
        }
        setNull();
    }
}
