public class Main {
    public static void main(String[] args) {
        /* Deadlock will occur if we use LockOne because it is not able
        to handle the case where both threads are trying to acquire the lock at the same time.
        */
//        runCounter(new Counter(new LockOne()),2);
        /* Starvation will occur in the absence of contention if we use LockTwo because thread alone
        can't enter the critical section if the other thread didn't make them as a victim.
         */
//        runCounter(new Counter(new LockTwo()),2);
        /* DeadLoack and Startvation can not occur*/
//        runCounter(new Counter(new PetersonLock()),2);
//        runCounter(new Counter(new FilterLock(4)),4);
        runCounter(new Counter(new BakeryLock(4)),4);
    }


    public static void runCounter(Counter counter, int noOfThreads) {
        // Create an array to hold threads
        Thread[] threads = new Thread[noOfThreads];

        for (int i = 0; i < noOfThreads; i++) {
            threads[i] = new Thread(
                    new Runnable() {
                        @Override
                        public void run() {
                            for (int j = 0; j < 1000; j++) {
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
