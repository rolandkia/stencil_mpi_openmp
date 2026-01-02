import subprocess
import re
import matplotlib.pyplot as plt
import numpy as np 

EXECUTABLE = "./stencil"  
TAILLES = [10, 16, 25, 32, 45, 64, 100, 128, 200, 256]

print(TAILLES)
def extraire_valeurs(texte):
    """Extrait la taille et les gflops du texte de sortie."""
    # Cherche "Taille: XXX x XXX" et capture le premier nombre
    taille_match = re.search(r"Taille:\s+(\d+)", texte)
    # Cherche "gflops = XXX"
    gflops_match = re.search(r"gflops\s+=\s+([\d.]+)", texte)
    
    if taille_match and gflops_match:
        return int(taille_match.group(1)), float(gflops_match.group(1))
    return None, None

def lancer_benchmark():
    tailles_reelles = []
    gflops_liste = []

    print(f"{'Taille (N)':>10} | {'GFLOPS':>15}")
    print("-" * 30)

    for n in TAILLES:
        # Exécution du programme avec la taille n passée en argument
        process = subprocess.run([EXECUTABLE, str(n)], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        if process.returncode == 0:
            taille, gflops = extraire_valeurs(process.stdout)
            if taille and gflops:
                tailles_reelles.append(taille)
                gflops_liste.append(gflops)
                print(f"{taille:10d} | {gflops:15.5f}")
        else:
            print(f"Erreur à n={n}: {process.stderr}")

    return tailles_reelles, gflops_liste

tailles, gflops = lancer_benchmark()

if tailles:
    plt.figure(figsize=(10, 6))
    plt.plot(tailles, gflops, marker='s', linestyle='-', color='green', linewidth=2)
    plt.xlabel("Taille (N x N)")
    plt.ylabel("GFLOPS (Giga-opérations par seconde)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.show()