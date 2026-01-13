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

parser = argparse.ArgumentParser(description="Benchmark Weak Scaling MPI ou OpenMP")
parser.add_argument("-m", "--modes", choices=['mpi', 'omp', 'mpi_omp'], nargs='+', required=True, 
                    help="Modes à tester (ex: -m mpi, -m omp, -m mpi omp, -m mpi_omp mpi)")
parser.add_argument("-s", "--size", type=int, default= 256, help="Charge de travail pour chaque processus")
parser.add_argument("-p","--max_p", nargs='+', type=int, help="Sequence de coeurs à tester")
args = parser.parse_args()

BASE_N = args.size
ITERATIONS = 3

# Nombre de thread par processus MPI uniquement si utilisation --modes mpi_omp
THREADS_FIXED = 6

if args.max_p == None:
	PROCS = [1, 2, 4, 6, 8, 10, 12]
else:
    PROCS = args.max_p
    
if 'mpi_omp' in args.modes:
    PROCS = [6, 12, 18, 24]



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
        env["OMP_PROC_BIND"] = "close"
        env["OMP_PLACES"] = "cores"
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
	print(f"Démarrage Weak Scaling - Mode: {mode}")
	print(f"{'P/Threads':>10} | {'N Total':>8} | {'GFLOPS Moy':>12} | {'Efficacité':>10}")
	print("-" * 50)

	results_gflops = []
	for p in PROCS:
		n_total = int(BASE_N * np.sqrt(p))
		avg_gflops = run_test(mode, p, n_total)
		results_gflops.append(avg_gflops)
		
		efficacite = (avg_gflops / (p * results_gflops[0])) * 100 if results_gflops[0] > 0 else 0
		print(f"{p:10d} | {n_total:8d} | {avg_gflops:12.4f} | {efficacite:9.1f}%")

	all_results[mode] = results_gflops

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)

for mode, gflops in all_results.items():
    eff_list = [(g / ((p/PROCS[0]) * gflops[0])) * 100 for g, p in zip(gflops, PROCS)]
    
    plt.plot(PROCS, eff_list, 'o-', label=f"Mode {mode.upper()}")


plt.axhline(y=100, color='black', linestyle='--', alpha=0.3)
plt.title(f'Efficacité Parallèle (Weak Scaling)\nCharge/coeur: {BASE_N}x{BASE_N}')
plt.xlabel("Nombre de processus/Threads")
plt.ylabel("Efficacité (%)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)


plt.subplot(1, 2, 2)
for mode, gflops in all_results.items():
    plt.plot(PROCS, gflops, 's-', label=f"Mode {mode.upper()}")

plt.title('Performance Totale')
plt.xlabel("Nombre de processus/Threads")
plt.ylabel("GFLOPS")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)


plt.tight_layout()
plt.savefig(f"benchmark_weak_scaling_{'_'.join(args.modes)}.png")
plt.show()
