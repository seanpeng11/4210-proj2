#include <omp.h>
#include "combined.h"
#include "../omp/gtmp.h"
#include "../mpi/gtmpi.h"

void combined_init(int num_threads, int num_processes) {
    gtmp_init(num_threads);
    gtmpi_init(num_processes);
}

void combined_barrier(void) {
    gtmp_barrier();
    if (omp_get_thread_num() == 0) {
        gtmpi_barrier();
    }
    gtmp_barrier();
}

void combined_finalize(void) {
    gtmpi_finalize();
    gtmp_finalize();
}
