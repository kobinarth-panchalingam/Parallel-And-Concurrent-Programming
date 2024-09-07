import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;

public class FilterLock implements Lock {
    private int noOfThreads;
    private volatile int[] level;
    private volatile int[] victim;
    public FilterLock(int n) {
        noOfThreads = n;
        level = new int[n];
        victim = new int[n];
        for (int i = 0; i < n; i++) {
            level[i] = 0;
        }
    }
    @Override
    public void lock() {
        int i = (int) (Thread.currentThread().getId() % noOfThreads);
        for (int l = 1; l < noOfThreads; l++) {
            level[i] = l;
            victim[l] = i;
            boolean conflict_exist = true;
            while (conflict_exist) {
                conflict_exist = false;
                for (int k = 0; k < noOfThreads; k++) {
                    if (k != i && level[k] >= l && victim[l] == i) {
                        conflict_exist = true;
                        break;
                    }
                }
            }
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
        int i = (int) (Thread.currentThread().getId() % noOfThreads);
        level[i] = 0;
    }

    @Override
    public Condition newCondition() {
        return null;
    }
}
