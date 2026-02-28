#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include "gtmpi.h"

int main(int argc, char** argv)
{
  int num_processes;
  int num_iter = 10;
  int warmup_iter = 0;
  int trials = 1;
  int rank;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (argc < 2){
    if (rank == 0) {
      fprintf(stderr, "Usage: ./harness [NUM_PROCS] [ITERS] [WARMUP] [TRIALS]\n");
    }
    MPI_Finalize();
    exit(EXIT_FAILURE);
  }

  num_processes = strtol(argv[1], NULL, 10);

  if (argc >= 5) {
    num_iter = strtol(argv[2], NULL, 10);
    warmup_iter = strtol(argv[3], NULL, 10);
    trials = strtol(argv[4], NULL, 10);
  }

  gtmpi_init(num_processes);

  if (rank == 0) {
    printf("procs,trial,us_per_barrier\n");
  }

  for (int trial = 0; trial < trials; trial++) {
    for (int i = 0; i < warmup_iter; i++) {
      gtmpi_barrier();
    }

    double start = MPI_Wtime();

    for (int i = 0; i < num_iter; i++) {
      gtmpi_barrier();
    }

    double end = MPI_Wtime();
    double elapsed = end - start;
    if (rank == 0) {
      printf("%d,%d,%.6f\n", num_processes, trial + 1, elapsed * 1e6 / num_iter);
    }
  }

  gtmpi_finalize();  

  MPI_Finalize();

  return 0;
}
