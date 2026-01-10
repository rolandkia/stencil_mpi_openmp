import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

# Configuration des arguments
parser = argparse.ArgumentParser(description="Benchmark de stencil (Séquentiel, MPI ou OpenMP)")
parser.add_argument("executable", help="Chemin vers l'exécutable")
parser.add_argument("-n", "--np", type=int, default=1, help="Nombre de processus MPI")
parser.add_argument("-nt", "--nthreads", type=int, default=1, help="Nombre de threads OpenMP")
parser.add_argument("-i", "--iterations", type=int, default=3, help="Nombre de répétitions")
args = parser.parse_args()

TAILLES = [400, 500]

def extraire_valeurs(texte):
    taille_match = re.search(r"Taille:\s+(\d+)", texte)
    gflops_match = re.search(r"gflops\s+=\s+([\d.]+)", texte)
    if taille_match and gflops_match:
        return int(taille_match.group(1)), float(gflops_match.group(1))
    return None, None

def lancer_benchmark():
    tailles_reelles = []
    gflops_moyens = []

    # Préparation de l'environnement pour OpenMP
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(args.nthreads)

    mode = "Séquentiel"
    if args.np > 1 and args.nthreads > 1:
        mode = f"Hybride (MPI:{args.np} x OMP:{args.nthreads})"
    elif args.np > 1:
        mode = f"MPI ({args.np} processus)"
    elif args.nthreads > 1:
        mode = f"OpenMP ({args.nthreads} threads)"

    print(f"Lancement du benchmark - Mode: {mode}")
    print(f"{'Taille (N)':>10} | {'GFLOPS Moy':>15} | {'Std Dev':>10}")
    print("-" * 45)

    for n in TAILLES:
        resultats_n = []
        taille_detectee = n

        for i in range(args.iterations):
            # Construction de la commande
            commande = []
            
            # Si MPI est demandé (soit par l'argument -n, soit par le nom du fichier)
            if args.np > 1 or "mpi" in args.executable.lower():
                commande = ["mpirun", "--oversubscribe", "-np", str(args.np)]
            
            commande.extend([args.executable, str(n)])

            # Exécution avec l'environnement configuré pour OpenMP
            process = subprocess.run(
                commande, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE, 
                text=True,
                env=env
            )
            
            if process.returncode == 0:
                taille, gflops = extraire_valeurs(process.stdout)
                if gflops:
                    resultats_n.append(gflops)
                    taille_detectee = taille
            else:
                print(f"Erreur à n={n}: {process.stderr}")

        if resultats_n:
            moyenne = np.mean(resultats_n)
            ecart_type = np.std(resultats_n)
            tailles_reelles.append(taille_detectee)
            gflops_moyens.append(moyenne)
            print(f"{taille_detectee:10d} | {moyenne:15.5f} | {ecart_type:10.2f}")

    return tailles_reelles, gflops_moyens

tailles, gflops = lancer_benchmark()

if tailles:
    plt.figure(figsize=(10, 6))
    label = f"{args.executable}"
    if args.np > 1: label += f" (MPI np={args.np})"
    if args.nthreads > 1: label += f" (OMP threads={args.nthreads})"
    
    plt.plot(tailles, gflops, marker='o', linestyle='-', label=label)
    plt.title(f"Benchmark Performance")
    plt.xlabel("Taille (N x N)")
    plt.ylabel("GFLOPS Moyens")
    plt.legend()
    plt.grid(True)
    plt.show()