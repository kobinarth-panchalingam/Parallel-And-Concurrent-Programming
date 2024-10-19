import java.util.Random;
import java.util.concurrent.Semaphore;

public class SenateBus {
    public static final long BUS_ARRIVAL_MEAN_TIME = 20 * 30 * 1000 / 20; // 20 minutes in milliseconds

    public static final long RIDER_ARRIVAL_MEAN_TIME = 30 * 1000 / 20; // 30 seconds in milliseconds

    public static int BUS_CAPACITY = 50;

    public static int waitingRiders = 0;

    public static Semaphore busArrived = new Semaphore(0);

    public static Semaphore busBoarded = new Semaphore(0);

    public static Semaphore riderMutex = new Semaphore(1);

    public static Semaphore busMutex = new Semaphore(1);

    public static Random random = new Random();

    public static void main(String[] args) {
        // Start a thread to generate buses and riders continuously
        Thread riderGenerator = new Thread(() -> {
            while (true) {
                Rider rider = new Rider();
                rider.start();

                try {
                    Thread.sleep(exponentiallyDistributedTime(RIDER_ARRIVAL_MEAN_TIME));
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });

        Thread busGenerator = new Thread(() -> {
            while (true) {
                Bus bus = new Bus();
                bus.start();

                // Sleep for an exponentially distributed time (mean 20 minutes)
                try {
                    Thread.sleep(exponentiallyDistributedTime(BUS_ARRIVAL_MEAN_TIME));
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
        });

        // Start both the rider and bus generators
        riderGenerator.start();
        busGenerator.start();
    }

    // Method to generate an exponentially distributed random time
    public static long exponentiallyDistributedTime(double meanMillis) {
        double u = random.nextDouble();
        return (long) (-Math.log(1 - u) * meanMillis);
    }

    static class Rider extends Thread {
        @Override
        public void run() {
            try {
                // Increment the number of waiting riders
                riderMutex.acquire();
                waitingRiders++;
                riderMutex.release();

                System.out.println("Rider with thread id " + Thread.currentThread().getId() + " waiting, and "
                        + " Total waiting riders: " + waitingRiders);

                // Wait for the bus to arrive
                busArrived.acquire();

                // Board the bus
                riderMutex.acquire();
                waitingRiders--;
                riderMutex.release();

                System.out.println("Rider with thread id " + Thread.currentThread().getId() + " boarded");

                // Signal the bus that the rider has boarded
                busBoarded.release();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }

    static class Bus extends Thread {
        @Override
        public void run() {
            try {
                // Allow only one bus at a time to board riders
                // busMutex.acquire();

                if (waitingRiders > 0) {
                    System.out.println("Bus with thread id " + Thread.currentThread().getId() + " arrived while "
                            + waitingRiders + " riders waiting");

                    int noOfRidersToBoard = Math.min(waitingRiders, BUS_CAPACITY);

                    for (int i = 0; i < noOfRidersToBoard; i++) {
                        // Signal a rider to board
                        busArrived.release();
                    }

                    // Wait for all the riders to board
                    for (int i = 0; i < noOfRidersToBoard; i++) {
                        busBoarded.acquire();
                    }

                    System.out.println("Bus with thread id " + Thread.currentThread().getId() + " departed with "
                            + noOfRidersToBoard + " riders");

                } else {
                    System.out.println("Bus left as no riders were waiting");
                }

                // busMutex.release();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
