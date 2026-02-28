#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>
#include "gtmpi.h"

static int g_num_procs = 0;
static int num_rounds = 0;
static int rank = 0;

typedef struct {
    int *remote;
    int parity;
} proc_state;

static proc_state state;

void gtmpi_init(int num_processes){
    g_num_procs = num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    num_rounds = 0;

    int tmp = 1;
    while (tmp < num_processes) {
        num_rounds++;
        tmp <<= 1;
    }

    state.parity = 0;
    state.remote = malloc(sizeof(int) * num_rounds);

    for (int i = 0; i < num_rounds; i++) {
        state.remote[i] = ((rank + (1 << i)) % num_processes);
    }

}

void gtmpi_barrier(){
    int send_buf = 1;
    int recv_buf = 0;
    for (int i = 0; i < num_rounds; i++) {
        int dest = state.remote[i];
        int source = (rank - (1 << i) + g_num_procs) % g_num_procs;
        int tag = state.parity * num_rounds + i;

        MPI_Sendrecv(&send_buf, 1, MPI_INT, dest, tag,
                     &recv_buf, 1, MPI_INT, source, tag,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    state.parity = 1 - state.parity;

}

void gtmpi_finalize(){
    free(state.remote);
    state.remote = NULL;

}
