#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include "combined.h"

int main(int argc, char **argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, num_procs;
    int num_iter = 10;
    int warmup_iter = 0;
    int trials = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: ./combined [NUM_THREADS] [ITERS] [WARMUP] [TRIALS]\n");
        }
        MPI_Finalize();
        return 1;
    }

    int num_threads = strtol(argv[1], NULL, 10);

    if (argc >= 5) {
        num_iter = strtol(argv[2], NULL, 10);
        warmup_iter = strtol(argv[3], NULL, 10);
        trials = strtol(argv[4], NULL, 10);
    }

    omp_set_dynamic(0);
    if (omp_get_dynamic())
        printf("Warning: dynamic adjustment of threads has been set\n");

    omp_set_num_threads(num_threads);
    combined_init(num_threads, num_procs);

    if (rank == 0) {
        printf("procs,threads,trial,us_per_barrier\n");
    }

    for (int trial = 0; trial < trials; trial++) {
        double start = 0.0;
        double end = 0.0;

        #pragma omp parallel shared(start, end, warmup_iter, num_iter)
        {
            int tid = omp_get_thread_num();

            for (int i = 0; i < warmup_iter; i++) {
                combined_barrier();
            }

            if (tid == 0) {
                start = omp_get_wtime();
            }

            for (int i = 0; i < num_iter; i++) {
                combined_barrier();
            }

            if (tid == 0) {
                end = omp_get_wtime();
            }
        }

        double elapsed = end - start;

        if (rank == 0) {
            printf("%d,%d,%d,%.6f\n", num_procs, num_threads, trial + 1, elapsed * 1e6 / num_iter);
        }
    }

    combined_finalize();
    MPI_Finalize();
    return 0;
}
