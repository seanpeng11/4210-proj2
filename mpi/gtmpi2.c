#include <stdlib.h>
#include <mpi.h>
#include <stdio.h>
#include "gtmpi.h"

static int g_num_procs = 0;
static int rank = 0;

static int parent = -1;
static int left = -1;
static int right = -1;


void gtmpi_init(int num_processes){
    g_num_procs = num_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        parent = -1;
    } else {
        parent = (rank - 1) / 2;
    }

    left = 2 * rank + 1;
    if (left >= g_num_procs) {
        left = -1;
    }
    right = 2 * rank + 2;
    if(right >= g_num_procs) {
        right = -1;
    }
}

void gtmpi_barrier(){
    int buf = 1;
    if (left != -1) {
        MPI_Recv(&buf, 1, MPI_INT, left, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (right != -1) {
        MPI_Recv(&buf, 1, MPI_INT, right, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (parent != -1) {
        MPI_Send(&buf, 1, MPI_INT, parent, 0, MPI_COMM_WORLD);
        MPI_Recv(&buf, 1, MPI_INT, parent, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    if (left != -1) {
        MPI_Send(&buf, 1, MPI_INT, left, 1, MPI_COMM_WORLD);
    }
    if (right != -1) {
        MPI_Send(&buf, 1, MPI_INT, right, 1, MPI_COMM_WORLD);
    }
}

void gtmpi_finalize(){
    g_num_procs = 0;

}
