import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;

public class LockOne implements Lock {
    private volatile boolean[]  flag = new boolean[2];
    @Override
    public void lock() {
        int i = (int) Thread.currentThread().getId()%2;
        int j = 1 - i;
        flag[i] = true;
        while (flag[j]){
        }
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
        int i = (int) Thread.currentThread().getId()%2;
        flag[i] = false;
    }

    @Override
    public Condition newCondition() {
        return null;
    }
}
