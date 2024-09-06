import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;

public class LockTwo implements Lock {
    private volatile int victim;
    @Override
    public void lock() {
        int i = (int) Thread.currentThread().getId()%2;
        victim = i;
        while (victim == i) {}

    }

    @Override
    public void lockInterruptibly() throws InterruptedException {

    }

    @Override
    public boolean tryLock() {
        return false;
    }

    @Override
    public boolean tryLock(long time, TimeUnit unit) throws InterruptedException {
        return false;
    }

    @Override
    public void unlock() {
    }

    @Override
    public Condition newCondition() {
        return null;
    }
}
