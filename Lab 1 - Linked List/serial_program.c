#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>

#define MAX 65536

struct node {
    int data;
    struct node* next;
};

// Global variables
struct node* head_p = NULL;
int no_of_operations;
int no_of_member_per_thread;
int no_of_insert_per_thread;
int no_of_delete_per_thread;

// Function prototypes
int insert(int value);
int member(int value);
int delete(int value);
void init(int no_of_variables);
void init_operations(int* operations);
long get_time();
void perform_operations(int* operations);
void free_list();

// Main function
int main(int argc, char* argv[]) {
    // check no of arguments
    if (argc != 6) {
        printf("Invalid number of arguments\n");
        return -1;
    }

    // Variables for time calculation
    long start, finish, elapsed;

    // Collecting arguments
    int no_of_variables = atoi(argv[1]);
    no_of_operations = atoi(argv[2]);
    no_of_member_per_thread = strtod(argv[3], NULL) * no_of_operations;
    no_of_insert_per_thread = strtod(argv[4], NULL) * no_of_operations;
    no_of_delete_per_thread = strtod(argv[5], NULL) * no_of_operations;

    // Initialize the linked list
    init(no_of_variables);

    // Initalize array of operations in random order
    int operations[no_of_operations];
    init_operations(operations);

    // Get the start time
    start = get_time();

    // Perform the operations
    perform_operations(operations);

    // Get the finish time
    finish = get_time();

    // Calculate the elapsed time
    elapsed = finish - start;

    // Free the linked list
    free_list();

    // Print the elapsed time
    printf("%ld\n", elapsed);
    return 0;

}

// Function to perform the operations in the operation array
void perform_operations(int* operations) {
    long i = 0;
    for (i = 0; i < no_of_operations; i++) {
        int value = rand() % MAX;
        if (operations[i] == 1) {
            insert(value);
        } else if (operations[i] == -1) {
            delete(value);
        } else {
            member(value);
        }
    }
}

// Function to get the current time in milliseconds
long get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000LL + tv.tv_usec / 1000;
}

// Function to initialize the linked list
void init(int no_of_variables) {
    srand(time(NULL));
    int i = 0;
    for (i = 0; i < no_of_variables; i++) {
        int value = rand() % MAX;
        int result = insert(value);
        if (result == 0) {
            i--;
        }
    }
}

// Function to initialize the array of operations
void init_operations(int* operations) {
    int i = 0;
    for (i = 0; i < no_of_operations; i++) {
        if (i < no_of_insert_per_thread) {
            operations[i] = 1;
        } else if (i < no_of_insert_per_thread + no_of_delete_per_thread) {
            operations[i] = -1;
        } else {
            operations[i] = 0;
        }
    }

    // Shuffle the array of operations using a random seed
    srand(time(NULL));
    for (i = no_of_operations - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = operations[i];
        operations[i] = operations[j];
        operations[j] = temp;
    }
}
// Function to free the linked list
void free_list() {
    struct node* curr_p;
    struct node* temp_p;

    if (head_p != NULL) {
        curr_p = head_p->next;
        head_p->next = NULL;
        while (curr_p != NULL) {
            temp_p = curr_p->next;
            free(curr_p);
            curr_p = temp_p;
        }
    }
}

// Function to insert a value into the linked list
int insert(int value) {
    struct node* curr_p = head_p;
    struct node* pred_p = NULL;
    struct node* temp_p;

    // Traverse the linked list to find the correct position to insert the value
    while (curr_p != NULL && curr_p->data < value) {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    // If the value is not present in the linked list, insert the value
    if (curr_p == NULL || curr_p->data > value) {
        temp_p = malloc(sizeof(struct node));
        temp_p->data = value;
        temp_p->next = curr_p;
        if (pred_p == NULL) {
            head_p = temp_p;
        } else {
            pred_p->next = temp_p;
        }
        return 1;
    } else {
        return 0;
    }
}

// Function to check if a value is present in the linked list
int member(int value) {
    struct node* curr_p = head_p;

    // Traverse the linked list to find the value
    while (curr_p != NULL && curr_p->data < value) {
        curr_p = curr_p->next;
    }
    
    if (curr_p == NULL || curr_p->data > value) {
        return 0;
    } else {
        return 1;
    }
}

// Function to delete a value from the linked list
int delete(int value) {
    struct node* curr_p = head_p;
    struct node* pred_p = NULL;

    // Traverse the linked list to find the value
    while (curr_p != NULL && curr_p->data < value) {
        pred_p = curr_p;
        curr_p = curr_p->next;
    }

    if (curr_p != NULL && curr_p->data == value) {
        if(pred_p == NULL) {
            head_p = curr_p->next;
            free(curr_p);
        } else {
            pred_p->next = curr_p->next;
            free(curr_p);
        }
        return 1;
    } else {
        return 0;
    }
}