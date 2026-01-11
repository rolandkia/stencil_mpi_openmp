#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static int rank, nb_procs;
static int local_size_y;

#define STENCIL_SIZE 25

typedef float stencil_t;

/** conduction coeff used in computation */
static const stencil_t alpha = 0.02;

/** threshold for convergence */
static const stencil_t epsilon = 0.0001;

/** max number of steps */
static const int stencil_max_steps = 100000;

static stencil_t*values = NULL;
static stencil_t*prev_values = NULL;

static int size_x = STENCIL_SIZE;
static int size_y = STENCIL_SIZE;

static void stencil_init_mpi(void){


	local_size_y = size_y/nb_procs; // on suppose dans un premier temps que size_y%nb_procs == 0

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

static void exchange_halos(void) {
    int up = (rank == 0) ? MPI_PROC_NULL : rank - 1;
    int down = (rank == nb_procs - 1) ? MPI_PROC_NULL : rank + 1;

    // Envoyer la 1ère ligne de données réelles vers le haut, recevoir dans le halo du bas
    MPI_Sendrecv(&values[size_x], size_x, MPI_FLOAT, up, 0,
                 &values[(local_size_y + 1) * size_x], size_x, MPI_FLOAT, down, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Envoyer la dernière ligne de données réelles vers le bas, recevoir dans le halo du haut
    MPI_Sendrecv(&values[local_size_y * size_x], size_x, MPI_FLOAT, down, 1,
                 &values[0], size_x, MPI_FLOAT, up, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}


static int stencil_step_mpi(void) {
    int local_convergence = 1;
    stencil_t* tmp = prev_values; 
	prev_values = values; 
	values = tmp;

	#pragma omp parallel for reduction(&&:local_convergence) schedule(static)
    // On ne calcule que de la ligne 1 à local_size_y (les lignes réelles)
    for (int y = 1; y <= local_size_y; y++) {
        // Ignorer les lignes de bordure physique tout en haut et tout en bas
        if ((rank == 0 && y == 1) || (rank == nb_procs - 1 && y == local_size_y)) continue;

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


static void stencil_free(void)
{
  free(values);
  free(prev_values);
}

/** display a (part of) the stencil values */
static void stencil_display(void){

	for (int p = 0; p < nb_procs; p++) {
		if (rank == p) {
			printf("--- Rang %d (Lignes locales: %d) ---\n", rank, local_size_y);
			// On affiche tout, y compris les halos (ligne 0 et local_size_y + 1)
			for (int y = 0; y < local_size_y + 2; y++) {
				printf("Ligne %d: ", y);
				for (int x = 0; x < size_x; x++) {
					printf("%6.2f ", values[x + size_x * y]);
				}
				printf("\n");
			}
			fflush(stdout); // Force l'affichage immédiat
		}
    	MPI_Barrier(MPI_COMM_WORLD); // Attend que le processus actuel ait fini d'écrire
	}
}


int main(int argc, char**argv)
{	
	MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);


	if (argc < 2) {
        fprintf(stderr, "Usage: %s <taille> ou %s <largeur> <hauteur>\n", argv[0], argv[0]);
    } else {
        size_x = atoi(argv[1]);
        size_y = (argc > 2) ? atoi(argv[2]) : size_x;
    }

    if (size_x <= 0 || size_y <= 0) {
        fprintf(stderr, "Erreur: Les dimensions doivent être positives.\n");
        return 1;
    }

	if (rank == 0){
		printf("# Taille: %d x %d\n", size_x, size_y);
  		printf("# init:\n");
	}


  	stencil_init_mpi();

	// exchange_halos();
	// stencil_display();
  
	double t1 = MPI_Wtime();
    int s, global_conv = 0;

    for (s = 0; s < stencil_max_steps; s++) {
        exchange_halos();
        int local_conv = stencil_step_mpi();

        MPI_Allreduce(&local_conv, &global_conv, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
        if (global_conv) break;
    }

    double t2 = MPI_Wtime();
    if (rank == 0) {
  		printf("# steps = %d\n", s);
		printf("# time = %g usecs.\n", t2 - t1);
        printf("# gflops = %g\n", (6.0 * size_x * size_y * s) / ((t2 - t1) * 1e9));
    }

    stencil_free();
    MPI_Finalize();
    return 0;
}
