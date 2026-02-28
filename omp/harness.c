#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "gtmp.h"

int main(int argc, char** argv)
{
  int num_threads;
  int num_iter = 10;
  int warmup_iter = 0;
  int trials = 1;

  if (argc < 2){
    fprintf(stderr, "Usage: ./harness [NUM_THREADS] [ITERS] [WARMUP] [TRIALS]\n");
    exit(EXIT_FAILURE);
  }
  num_threads = strtol(argv[1], NULL, 10);

  if (argc >= 5) {
    num_iter = strtol(argv[2], NULL, 10);
    warmup_iter = strtol(argv[3], NULL, 10);
    trials = strtol(argv[4], NULL, 10);
  }

  omp_set_dynamic(0);
  if (omp_get_dynamic())
    printf("Warning: dynamic adjustment of threads has been set\n");

  omp_set_proc_bind(omp_proc_bind_true);
  omp_set_num_threads(num_threads);
  
  gtmp_init(num_threads);

  printf("threads,trial,us_per_barrier\n");
  for (int trial = 0; trial < trials; trial++) {
    double start = 0.0;
    double end = 0.0;

#pragma omp parallel shared(start, end, warmup_iter, num_iter)
    {
      int tid = omp_get_thread_num();

      for (int i = 0; i < warmup_iter; i++) {
        gtmp_barrier();
      }

      if (tid == 0) {
        start = omp_get_wtime();
      }

      for (int i = 0; i < num_iter; i++) {
        gtmp_barrier();
      }

      if (tid == 0) {
        end = omp_get_wtime();
      }
    }

    printf("%d,%d,%.6f\n", num_threads, trial + 1, (end - start) * 1e6 / num_iter);
  }

  gtmp_finalize();

  return 0;
}
