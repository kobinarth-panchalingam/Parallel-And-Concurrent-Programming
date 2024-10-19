import java.util.concurrent.locks.Lock;

public class Counter {
    private int count = 0;
    private Lock lock;

    public Counter(Lock lock) {
        this.lock = lock;
    }

    public int getAndIncrement() {
        lock.lock();
        try {
            int temp = count;
            count = temp + 1;
            return temp;
        } finally {
            lock.unlock();
        }
    }
}
