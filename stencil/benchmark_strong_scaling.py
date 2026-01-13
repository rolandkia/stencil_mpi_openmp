import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

executables = {
    "mpi": "./stencil_mpi", 
	"omp" : "./stencil_openmp",
	"mpi_omp" : "./stencil_mpi_openmp"
}
parser = argparse.ArgumentParser(description="Benchmark Strong Scaling MPI ou OpenMP")
parser.add_argument("-m", "--modes", choices=['mpi', 'omp', 'mpi_omp'], nargs='+', required=True, 
                    help="Modes à tester (ex: -m mpi, -m omp, -m mpi omp, -m mpi_omp mpi)")
parser.add_argument("-s", "--size", type=int, default= 256, help="Charge de travail commune")
parser.add_argument("-p","--max_p", nargs='+', type=int, help="Sequence de coeurs à tester")
args = parser.parse_args()

N_FIXE = args.size
ITERATIONS = 3


# Nombre de thread par processus MPI uniquement si utilisation --modes mpi_omp
THREADS_FIXED = 6

if args.max_p == None:
	PROCS = [1, 2, 4, 6, 8, 10, 12]
elif 'mpi_omp' in args.modes:
    PROCS = [6, 12, 18, 24]
else:
    PROCS = args.max_p


def run_test(mode, p_val, size):
    EXEC = executables[mode]
    env = os.environ.copy()
    
    if mode == 'mpi':
        # Commande MPI
        cmd = [
            "mpirun", "-np", str(p_val),
            "--bind-to", "core",
            EXEC, str(size)
        ]
    elif mode == 'omp':
		# Commande OpenMP
        env["OMP_NUM_THREADS"] = str(p_val)
        cmd = [EXEC, str(size)]
        
    else:
		# Commande MPI_OpenMP
        env["OMP_NUM_THREADS"] = str(THREADS_FIXED)
        env["OMP_PROC_BIND"] = "true"
        env["OMP_PLACES"] = "cores"
        cmd = [
            "mpirun", "-np", str(p_val//THREADS_FIXED),
            "--map-by", f"numa:PE={THREADS_FIXED}",
            "--bind-to", "core",
            EXEC, str(size)]  
    
    gflops_list = []
    for i in range(ITERATIONS):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            match = re.search(r"gflops\s+=\s+([\d.]+)", result.stdout)
            if match:
                gflops_list.append(float(match.group(1)))
        except Exception as e:
            print(f"Erreur système: {e}")
            
    return np.mean(gflops_list) if gflops_list else 0


all_results = {}

for mode in args.modes:
	print(f"Démarrage Strong Scaling - Mode: {mode}")
	print(f"{'P/Threads':>10} | {'GFLOPS Moy':>12} |{'SpeedUp':>10}")
	print("-" * 50)

	results_gflops = []
	for p in PROCS:
		avg_gflops = run_test(mode, p, N_FIXE)
		results_gflops.append(avg_gflops)
		
		speedup = avg_gflops / results_gflops[0]
		
		print(f"{p:4d} | {avg_gflops:12.4f} | {speedup:10.2f}")

	all_results[mode] = results_gflops

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

for mode, gflops in all_results.items():
	speedup_reel = [g / gflops[0] for g in gflops]
	plt.plot(PROCS, speedup_reel, 'o-', label=f"Mode {mode}")


plt.plot(PROCS, PROCS, '--', label='Ideal', color='gray', alpha=0.7)
plt.xlabel("Nombre de processus/Threads")
plt.ylabel('Speedup (T1/Tp)')
plt.title(f'Scalabilité Forte (Speedup)\nGrille {N_FIXE}x{N_FIXE}')
plt.legend()
plt.grid(True, linestyle=':')


plt.subplot(1, 2, 2)

for mode, gflops in all_results.items():
    plt.plot(PROCS, gflops, 's-', label=f"Mode {mode}")

plt.title('Performance Totale')
plt.xlabel("Nombre de processus/Threads")
plt.ylabel("GFLOPS")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)


plt.tight_layout()
plt.savefig(f"benchmark_strong_scaling_{'_'.join(args.modes)}.png")
plt.show()




