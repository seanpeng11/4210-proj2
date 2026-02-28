#include <omp.h>
#include <stdlib.h>
#include <stdbool.h>
#include "gtmp.h"

static int g_num_threads = 0;
static int num_rounds = 0;

typedef struct {
    volatile bool *local;
    volatile bool **remote;
    volatile bool sense;
    int parity;
} thread_state;

static thread_state *states = NULL;

static int idx(int parity, int round) {
    return parity * num_rounds + round;
}


void gtmp_init(int num_threads){
    g_num_threads = num_threads;
    num_rounds = 0;
    int tmp = 1;
    while (tmp < num_threads) {
        num_rounds++;
        tmp <<= 1;
    }

    states = malloc(sizeof(thread_state) * num_threads);

    for (int i = 0; i < num_threads; i++) {
        states[i].sense = true;
        states[i].parity = 0;
        states[i].local  = malloc(sizeof(bool) * 2 * num_rounds);
        states[i].remote = malloc(sizeof(bool*) * 2 * num_rounds);
    }
    for (int i = 0; i < num_threads; i++) {
        for (int p = 0; p < 2; p++) {
            for (int r = 0; r < num_rounds; r++) {
                int j = idx(p, r);
                states[i].local[j] = false;

                int partner = (i + (1 << r)) % g_num_threads;
                states[i].remote[j] = &states[partner].local[j];
            }
        }
    }
}

void gtmp_barrier(){
    int i = omp_get_thread_num();
    for (int r = 0; r < num_rounds; r++) {
        int j = idx(states[i].parity, r);
        *states[i].remote[j] = states[i].sense;
        while (states[i].local[j] != states[i].sense);
    }

    if (states[i].parity == 1) {
        states[i].sense = !states[i].sense;
    }
    states[i].parity = 1 - states[i].parity;

}

void gtmp_finalize(){
    if (states != NULL) {
        for (int i = 0; i < g_num_threads; i++) {
            free((void *)states[i].local);
            free((void *)states[i].remote);
        }
        free(states);
        states = NULL;
    }
}
