CC      = gcc
MPICC   = mpicc
CFLAGS += -Wall -g -O4 -fopenmp -std=gnu11
LDLIBS += -lm -lrt


TARGETS = stencil stencil_mpi stencil_openmp stencil_mpi_openmp


all: $(TARGETS)

stencil: stencil.c
	$(CC) $(CFLAGS) stencil.c -o stencil $(LDLIBS)


stencil_mpi: stencil_mpi.c
	$(MPICC) $(CFLAGS) stencil_mpi.c -o stencil_mpi $(LDLIBS)


stencil_openmp: stencil_openmp.c
	$(CC) $(CFLAGS) stencil_openmp.c -o stencil_openmp $(LDLIBS)

stencil_mpi_openmp: stencil_mpi_openmp.c
	$(MPICC) $(CFLAGS) stencil_mpi_openmp.c -o stencil_mpi_openmp $(LDLIBS)


clean:
	-rm -f $(TARGETS)

mrproper: clean
	-rm -f *~

archive:
	( cd .. ; tar czf stencil.tar.gz stencil/ )

.PHONY: all clean mrproper archive