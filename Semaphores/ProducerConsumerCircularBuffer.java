import java.util.concurrent.Semaphore;

public class ProducerConsumerCircularBuffer {
    public static int [] buffer = new int[10];
    public static int i = 0;
    public static int head = 0;
    public static int tail = 0;
    public static Semaphore elements = new Semaphore(0);
    public static Semaphore spaces = new Semaphore(10);
    public static Semaphore mutex = new Semaphore(1);

    static class Producer extends Thread {
        public void run() {
            while (i < 30){
                try {
                    spaces.acquire();

                    mutex.acquire();

                    int index = tail % 10;
                    buffer[index] = i;
                    System.out.println("Producer produced " + i + " tail " + index);

                    i++;
                    tail ++;

                    mutex.release();

                    elements.release();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }

            }
        }
    }

    static class Consumer extends Thread {
        public void run() {
            while (true) {
                try {
                    elements.acquire();

                    mutex.acquire();

                    int index = head % 10;
                    int i = buffer[index];
                    System.out.println("Consumer aquired " + i + " head " + index);

                    head ++;

                    mutex.release();

                    spaces.release();
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    public static void main(String[] args) {
        Producer producer1 = new Producer();
        Producer producer2 = new Producer();
        Consumer consumer1 = new Consumer();
        Consumer consumer2 = new Consumer();

        consumer1.start();
        consumer2.start();
        producer1.start();
        producer2.start();
    }
}
