import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;

public class BakeryLock implements Lock {
    private volatile int noOfThreads;
    private boolean[] flag;
    private int[] label;
    public BakeryLock(int n) {
        noOfThreads = n;
        flag = new boolean[n];
        label = new int[n];
        for (int i = 0; i < n; i++) {
            flag[i] = false;
            label[i] = 0;
        }
    }

    @Override
    public void lock() {
        int i = (int) (Thread.currentThread().getId() % noOfThreads);
        flag[i] = true;
        int max = 0;
        for (int j = 0; j < noOfThreads; j++) {
            if (label[j] > max) {
                max = label[j];
            }
        }
        label[i] = max + 1;
        boolean wait = true;
        while (wait) {
            wait = false;
            for (int k = 0; k < noOfThreads; k++) {
                if (k != i && flag[k] && (label[k] < label[i] || (label[k] == label[i] && k < i))) {
                    wait = true;
                    break;
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
        flag[i] = false;
    }

    @Override
    public Condition newCondition() {
        return null;
    }
}
