package br.com.guialves.rflr.utils.datastructure;

import java.util.*;

import static java.util.Objects.checkIndex;
import static java.util.Objects.requireNonNull;

public final class RingBuffer<E> extends AbstractQueue<E> {
    private final Object[] buffer;
    private int head;
    private int size;
    private int modCount;

    public RingBuffer(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("capacity must be > 0: " + capacity);
        }
        buffer = new Object[capacity];
    }

    public int capacity() {
        return buffer.length;
    }

    @Override
    public boolean offer(E element) {
        requireNonNull(element);
        int tail = (head + size) % buffer.length;
        buffer[tail] = element;
        if (size < buffer.length) {
            size++;
        } else {
            head = (head + 1) % buffer.length;
        }
        modCount++;
        return true;
    }

    @SuppressWarnings("unchecked")
    @Override
    public E poll() {
        if (size == 0) return null;
        E e = (E) buffer[head];
        buffer[head] = null;
        head = (head + 1) % buffer.length;
        size--;
        modCount++;
        return e;
    }

    @SuppressWarnings("unchecked")
    @Override
    public E peek() {
        return size == 0 ? null : (E) buffer[head];
    }

    @SuppressWarnings("unchecked")
    public E get(int i) {
        checkIndex(i, size);
        return (E) buffer[(head + i) % buffer.length];
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public void clear() {
        Arrays.fill(buffer, null);
        head = 0;
        size = 0;
        modCount++;
    }

    @Override
    public Iterator<E> iterator() {
        return new Iterator<>() {
            private int next = 0;
            private final int expectedModCount = modCount;

            @Override
            public boolean hasNext() {
                return next < size;
            }

            @Override
            public E next() {
                if (modCount != expectedModCount) throw new ConcurrentModificationException();
                if (next >= size) throw new NoSuchElementException();
                return get(next++);
            }
        };
    }
}
