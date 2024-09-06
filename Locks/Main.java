public class Main {
    public static void main(String[] args) {
        runCounter();
    }

    public static void runCounter() {
        Counter counter = new Counter();

        // No of threads
        int noOfThreads = 2;

        // Create an array to hold threads
        Thread[] threads = new Thread[noOfThreads];

        for (int i = 0; i < noOfThreads; i++) {
            threads[i] = new Thread(
                    new Runnable() {
                        @Override
                        public void run() {
                            for (int j = 0; j < 100; j++) {
                                System.out.println(Thread.currentThread().getName() + ": " + counter.getAndIncrement());
                            }
                        }
                    }
            );
        }

        // Start all thread
        for (int i = 0; i < noOfThreads; i++) {
            threads[i].start();
        }

        // Join all threads
        for (int i = 0; i < noOfThreads; i++) {
            try {
                threads[i].join();
            } catch (InterruptedException e) {
                throw new RuntimeException(e);
            }
        }

        // Print final value
        System.out.println("Final counter value: "+ counter.getAndIncrement());
    }
}
