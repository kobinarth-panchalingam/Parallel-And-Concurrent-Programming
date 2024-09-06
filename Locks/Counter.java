import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Counter {
    private int count = 0;
    private Lock lock = new LockOne();

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
