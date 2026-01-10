import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description="Benchmark Weak Scaling MPI ou OpenMP")
parser.add_argument("-m", "--mode", choices=['mpi', 'omp'], help="Mode d'exécution : mpi ou omp")
parser.add_argument("-s", "--size", type=int, default= 256, help="Charge de travail pour chaque processus")
parser.add_argument("executable", help="Chemin vers l'exécutable")
parser.add_argument("--max_p", type=int, default=12, help="Nombre max de coeurs à tester")
args = parser.parse_args()

EXEC = args.executable
BASE_N = args.size
PROCS = [1, 2, 4, 6, 8, 10, 12] # Ou list(range(1, args.max_p + 1))
ITERATIONS = 3

def run_test(p_val, size):
    env = os.environ.copy()
    
    if args.mode == 'mpi':
        # Commande MPI
        cmd = [
            "mpirun", "-np", str(p_val),
            "--mca", "pml", "^ucx", "--mca", "osc", "^ucx",
            "--bind-to", "core",
            EXEC, str(size)
        ]
    else:
        # Commande OpenMP
        env["OMP_NUM_THREADS"] = str(p_val)
        cmd = [EXEC, str(size)]
    
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

print(f"Démarrage Weak Scaling - Mode: {args.mode.upper()}")
print(f"{'P/Threads':>10} | {'N Total':>8} | {'GFLOPS Moy':>12} | {'Efficacité':>10}")
print("-" * 50)

results_gflops = []
for p in PROCS:
    n_total = int(BASE_N * np.sqrt(p))
    avg_gflops = run_test(p, n_total)
    results_gflops.append(avg_gflops)
    
    efficacite = (avg_gflops / (p * results_gflops[0])) * 100 if results_gflops[0] > 0 else 0
    print(f"{p:10d} | {n_total:8d} | {avg_gflops:12.4f} | {efficacite:9.1f}%")

plt.figure(figsize=(15, 6))

label_x = "Nombre de processus MPI" if args.mode == 'mpi' else "Nombre de threads OpenMP"

plt.subplot(1, 2, 1)
efficacite_list = [(g / (p * results_gflops[0])) * 100 for g, p in zip(results_gflops, PROCS)]
plt.plot(PROCS, efficacite_list, 'o-', color='red', linewidth=2)
plt.axhline(y=100, color='gray', linestyle='--')
plt.title(f'Efficacité Parallèle ({args.mode.upper()})\n Charge constante : {BASE_N}x{BASE_N} par coeur')
plt.xlabel(label_x)
plt.ylabel('Efficacité (%)')
plt.ylim(0, 110)
plt.grid(True, linestyle=':')

plt.subplot(1, 2, 2)
gflops_ideaux = [p * results_gflops[0] for p in PROCS]
plt.plot(PROCS, results_gflops, 's-', color='green', linewidth=2, label='Mesurés')
plt.plot(PROCS, gflops_ideaux, '--', color='gray', alpha=0.7, label='Idéal')
plt.title(f'Performance Totale\nMode {args.mode.upper()}')
plt.xlabel(label_x)
plt.ylabel('GFLOPS Globaux')
plt.legend()
plt.grid(True, linestyle=':')

plt.tight_layout()
plt.savefig(f"weak_scaling_{args.mode}.png")
plt.show()