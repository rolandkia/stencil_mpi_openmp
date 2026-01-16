import matplotlib.pyplot as plt
import numpy as np
import argparse
import subprocess
import os
import re

executables = {
	"seq" : "./stencil",
	"mpi" : "./stencil_mpi",
	"omp" : "./stencil_openmp",
	"mpi_omp" : "./stencil_mpi_openmp",
	"rec" : "./stencil_recouvrement"
}

parser = argparse.ArgumentParser(description="Benchmark Seq vs MPI vs OMP vs MPI + OMP")
parser.add_argument("-s", "--size", type=int, default= 512, help="Taille de la matrice")
args = parser.parse_args()

BASE_N = args.size
ITERATIONS = 3

def run_test(mode, size):
    EXEC = executables[mode]
    env = os.environ.copy()
    
    if mode == "seq":
        print("")
        cmd = [EXEC, str(size)]
        
    elif mode == "mpi":
        env["OMP_NUM_THREADS"] = str(1)
        cmd = [
            "mpirun", "-np", str(24),
            "--bind-to", "core",
            EXEC, str(size)]
    
    elif mode == "omp":
        env["OMP_NUM_THREADS"] = str(24)
        env["OMP_PROC_BIND"] = "close"
        env["OMP_PLACES"] = "cores"
        cmd = [EXEC, str(size)]
        
    elif mode == "mpi_omp" or mode == "rec":
        env["OMP_NUM_THREADS"] = str(6)
        env["OMP_PROC_BIND"] = "true"
        env["OMP_PLACES"] = "cores"
        cmd = [
            "mpirun", "-np", str(4),
            "--map-by", "numa:PE=6",
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

print(f"{'Mode':>10} | {'N Total':>8} | {'GFLOPS Moy':>12}")
print("-" * 30)

all_results = {}

for mode, exe in executables.items():
    
	avg_gflops = run_test(mode, BASE_N)
	print(f"{mode:>10s} | {BASE_N:8d} | {avg_gflops:12.4f} ")
	all_results[mode] = avg_gflops


data = {
	"Séquentiel": all_results["seq"],      # 1 cœur
    "Pur MPI": all_results["mpi"],         # 24 processus
    "Pur OpenMP": all_results["omp"],      # 24 threads
    "Hybride (4 MPI x 6 OMP)": all_results["mpi_omp"],   # 4 MPI x 6 OMP (avec binding)
    "Recouvrement (4 MPI x 6 OMP)" : all_results['rec']
}


labels = list(data.keys())
values = list(data.values())
colors = ['#95a5a6', '#3498db', '#e67e22', '#2ecc71', "#cf4a07"] # Gris, Bleu, Orange, Vert

plt.figure(figsize=(10, 7))
bars = plt.bar(labels, values, color=colors, edgecolor='black', alpha=0.8)

plt.title(f"Comparaison des Performances sur 1 Noeud (24 Coeurs)\n Taille {BASE_N}x{BASE_N}", 
          fontsize=14, pad=20)
plt.ylabel("GFLOPS", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.ylim(0, max(values) * 1.2) 

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, 
             f'{yval:.2f}', ha='center', va='bottom', fontsize=11)


plt.tight_layout()
plt.savefig("comparaison_modeles_24_coeurs.png", dpi=300)
plt.show()