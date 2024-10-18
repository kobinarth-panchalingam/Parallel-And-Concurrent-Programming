import java.util.ArrayList;
import java.util.concurrent.Semaphore;

public class ProducerConsumerInfiniteBuffer {
    public static ArrayList<Integer> buffer = new ArrayList<>();
    public static Integer i = 0;
    public static Integer head = 0;
    public static Integer tail = 0;
    public static Semaphore semaphore = new Semaphore(0);

    static class Producer extends Thread {
        public void run() {
            for (int j = 0; j < 1000; j++){
                buffer.add(tail,i);
                i++;
                tail ++;
                System.out.println("tail " + tail + " i " + i);
                semaphore.release();
            }
        }
    }

    static class Consumer extends Thread {
        public void run() {
            while (true) {
                try {
                    semaphore.acquire();
                    Integer i = buffer.get(head);
                    head ++;
                    System.out.println("Consumer aquired " + i);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }

    public static void main(String[] args) {
        Producer producer = new Producer();
        Consumer consumer = new Consumer();
        consumer.start();
        producer.start();
    }
}
