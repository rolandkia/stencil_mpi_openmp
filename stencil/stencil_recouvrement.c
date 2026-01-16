#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

static int rank, nb_procs;
static int local_size_y;

#define STENCIL_SIZE 25

typedef float stencil_t;

static const stencil_t alpha = 0.02;
static const stencil_t epsilon = 0.0001;
static const int stencil_max_steps = 100000;

static stencil_t*values = NULL;
static stencil_t*prev_values = NULL;

static int size_x = STENCIL_SIZE;
static int size_y = STENCIL_SIZE;

static void stencil_init_mpi(void){
	local_size_y = size_y/nb_procs;

	size_t mem_size = (local_size_y+2)*size_x*sizeof(stencil_t);
	values = malloc(mem_size);
	prev_values = malloc(mem_size);

	memset(values, 0, mem_size);

	if (rank == 0){
		for (int x = 0; x<size_x; x++){
			values[x + size_x * 0] = x;
			values[x + size_x * 1] = x;
		}
	}

	if (rank == nb_procs - 1){
		for (int x = 0; x<size_x; x++){
      		values[x + size_x * (local_size_y+1)] = size_x - x - 1;
      		values[x + size_x * (local_size_y+0)] = size_x - x - 1;
		}
	}

	for(int y_local = 1; y_local <= local_size_y; y_local++) {
		int y_global = (y_local - 1) + (rank * local_size_y);
    	values[0 + size_x * y_local] = y_global;
		values[(size_x - 1) + size_x * y_local] = size_y - y_global - 1;
	}

	memcpy(prev_values, values, mem_size);
}

static int stencil_step_hybrid_overlap(void) {
    int local_convergence = 1;
    stencil_t* tmp = prev_values; 
    prev_values = values; 
    values = tmp;

    MPI_Request requests[4];
    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == nb_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // ========== PHASE 1: LANCER LES COMMUNICATIONS ==========
    MPI_Irecv(&values[0], size_x, MPI_FLOAT, up, 1, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(&values[(local_size_y + 1) * size_x], size_x, MPI_FLOAT, down, 0, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(&prev_values[size_x], size_x, MPI_FLOAT, up, 0, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(&prev_values[local_size_y * size_x], size_x, MPI_FLOAT, down, 1, MPI_COMM_WORLD, &requests[3]);

    // ========== PHASE 2: CALCUL INTÉRIEUR (lignes 2 à local_size_y - 1) ==========
    // Ces lignes n'ont pas besoin des halos MPI
    #pragma omp parallel for reduction(&&:local_convergence) schedule(static)
    for (int y = 2; y <= local_size_y - 1; y++) {
        for (int x = 1; x < size_x - 1; x++) {
            values[x + size_x * y] = alpha * ( 
                prev_values[x - 1 + size_x * y] + prev_values[x + 1 + size_x * y] +
                prev_values[x + size_x * (y - 1)] + prev_values[x + size_x * (y + 1)]
            ) + (1.0 - 4.0 * alpha) * prev_values[x + size_x * y];

            if (local_convergence && (fabs(prev_values[x + size_x * y] - values[x + size_x * y]) > epsilon))
                local_convergence = 0;
        }
    }

    // ========== PHASE 3: ATTENDRE LES HALOS ==========
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);

    // CALCUL DES BORDURES (lignes qui touchent les halos MPI) ==========
    // Ligne supérieure (y=1) : seulement si on n'est pas le premier rang
    if (rank != 0) {
        int y = 1;
        #pragma omp parallel for reduction(&&:local_convergence) schedule(static)
        for (int x = 1; x < size_x - 1; x++) {
            values[x + size_x * y] = alpha * ( 
                prev_values[x - 1 + size_x * y] + prev_values[x + 1 + size_x * y] +
                prev_values[x + size_x * (y - 1)] + prev_values[x + size_x * (y + 1)]
            ) + (1.0 - 4.0 * alpha) * prev_values[x + size_x * y];

            if (local_convergence && (fabs(prev_values[x + size_x * y] - values[x + size_x * y]) > epsilon))
                local_convergence = 0;
        }
    }

    // Ligne inférieure (y=local_size_y) : seulement si on n'est pas le dernier rang
    if (rank != nb_procs - 1) {
        int y = local_size_y;
        #pragma omp parallel for reduction(&&:local_convergence) schedule(static)
        for (int x = 1; x < size_x - 1; x++) {
            values[x + size_x * y] = alpha * ( 
                prev_values[x - 1 + size_x * y] + prev_values[x + 1 + size_x * y] +
                prev_values[x + size_x * (y - 1)] + prev_values[x + size_x * (y + 1)]
            ) + (1.0 - 4.0 * alpha) * prev_values[x + size_x * y];

            if (local_convergence && (fabs(prev_values[x + size_x * y] - values[x + size_x * y]) > epsilon))
                local_convergence = 0;
        }
    }

    return local_convergence;
}

static void stencil_free(void) {
    free(values);
    free(prev_values);
}

int main(int argc, char**argv) {	
    // Initialisation avec support multi-threading
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s <taille> ou %s <largeur> <hauteur>\n", argv[0], argv[0]);
        MPI_Finalize();
        return 1;
    }

    size_x = atoi(argv[1]);
    size_y = (argc > 2) ? atoi(argv[2]) : size_x;

    if (size_x <= 0 || size_y <= 0) {
        if (rank == 0) fprintf(stderr, "Erreur: Les dimensions doivent être positives.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("# Taille: %d x %d\n", size_x, size_y);
        printf("# Processus MPI: %d\n", nb_procs);
        printf("# Threads OpenMP: %d\n", omp_get_max_threads());
    }

    stencil_init_mpi();
  
    double t1 = MPI_Wtime();
    int s, local_conv, global_conv = 0, prev_global_conv = 0;
    MPI_Request conv_req = MPI_REQUEST_NULL;

    for (s = 0; s < stencil_max_steps; s++) {
        // Calcul de l'itération s
        local_conv = stencil_step_hybrid_overlap(); 

        // Vérifier la convergence de l'itération s-1
        if (s > 0) {
            MPI_Wait(&conv_req, MPI_STATUS_IGNORE);
            if (prev_global_conv) {
                if (rank == 0) printf("# Convergence atteinte à l'itération %d\n", s - 1);
                break;
            }
        }

        // Lancer la réduction pour l'itération s (non-bloquante)
        MPI_Iallreduce(&local_conv, &global_conv, 1, MPI_INT, MPI_LAND, 
                       MPI_COMM_WORLD, &conv_req);
        
        prev_global_conv = global_conv;
    }

    // Attendre la dernière réduction
    if (conv_req != MPI_REQUEST_NULL) {
        MPI_Wait(&conv_req, MPI_STATUS_IGNORE);
    }

    double t2 = MPI_Wtime();
    
    if (rank == 0) {
        printf("# steps = %d\n", s);
        printf("# time = %g secs.\n", t2 - t1);
        printf("# gflops = %g\n", (6.0 * size_x * size_y * s) / ((t2 - t1) * 1e9));
    }

    stencil_free();
    MPI_Finalize();
    return 0;
}
