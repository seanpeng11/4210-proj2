#include <omp.h>
#include <stdlib.h>
#include <stdbool.h>
#include "gtmp.h"

static int g_num_threads = 0;
static int count = 0;
static bool sense = false;
static bool *my_sense = NULL;

void gtmp_init(int num_threads){
    g_num_threads = num_threads;
    count = num_threads;
    sense = false;

    my_sense = malloc(sizeof(bool) * num_threads);
    for (int i = 0; i < num_threads; i++) {
        my_sense[i] = false;
    }

}

void gtmp_barrier(){
    int i = omp_get_thread_num();
    int order;
    my_sense[i] = !my_sense[i];

    #pragma omp atomic capture
    order = --count;

    if (order == 0) {
        count = g_num_threads;
        sense = my_sense[i];
    } else {
        while (sense != my_sense[i]);
    }
}

void gtmp_finalize(){
    free(my_sense);
    my_sense = NULL;
}

