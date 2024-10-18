import java.util.concurrent.Semaphore;

public class MutualExclusion {
    public static Semaphore semaphore = new Semaphore(1);

    // Process 1
    static class Process1 extends Thread {
        public void run() {
            while (true) {
                try {
                    // Non-critical section
                    System.out.println("Process 1: In non-critical section");

                    // Down operation
                    semaphore.acquire();
                    System.out.println("Process 1: Entering Critical section");

                    // Critical section
                    Thread.sleep(1000);

                    System.out.println("Process 1: Exiting Critical section");
                    semaphore.release();

                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }

        }
    }

    static class Process2 extends Thread {
        public void run() {
            while (true) {
                try {
                    // Non-critical section
                    System.out.println("Process 2: In non-critical section");

                    // Down operation
                    semaphore.acquire();
                    System.out.println("Process 2: Entering Critical section");

                    // Critical section
                    Thread.sleep(1000);

                    System.out.println("Process 2: Exiting Critical section");
                    semaphore.release();
                    Thread.sleep(1000);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }

        }
    }

    public static void main(String[] args) {
        Process1 p1 = new Process1();
        Process2 p2 = new Process2();

        p1.start();
        p2.start();
    }
}
