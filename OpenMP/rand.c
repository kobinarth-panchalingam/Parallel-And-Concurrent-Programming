// C program for generating a
// random double number in a given range.
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Generates and prints 'count' random
// double numbers in range [min, max].
void printRandoms(double min, double max, int count) {
    printf("Random numbers between %.2f and %.2f: ", min, max);
  
    // Loop that will print the count random numbers
    for (int i = 0; i < count; i++) {

        // Find the random number in the range [min, max]
        double scale = rand() / (double) RAND_MAX; // [0, 1.0]
        double rd_num = min + scale * (max - min); // [min, max]

        printf("%.2f ", rd_num);
    }
}

int main() {
    srand(time(0)); // Seed the random number generator
    double min = -1.0, max = 1.0;
    int count = 10;
    printRandoms(min, max, count);
    return 0;
}

// gcc -o rand rand.c