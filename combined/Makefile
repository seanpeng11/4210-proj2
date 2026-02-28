MPICC = mpicc.mpich

OMPFLAGS = -fopenmp
OMPLIBS = -lgomp

CFLAGS = -g -std=gnu99 -I. -I../omp -I../mpi -Wall $(OMPFLAGS)
LDLIBS = $(OMPLIBS) -lm

all: combined 

SRCS := harness.c combined.c ../omp/gtmp1.c ../mpi/gtmpi2.c
OBJS = $(SRCS:.c=.o)

%.o: %.c
	$(MPICC) -c $(CFLAGS) $< -o $@

combined: $(OBJS) 
	$(MPICC) -o $@ $(CFLAGS) $(LDFLAGS) $^ $(LDLIBS)

clean:
	rm -rf *.o ../omp/*.o ../mpi/*.o *.dSYM combined

