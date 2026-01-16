# Stencil MPI+OpenMP

This project implements a 2D stencil computation with different parallelization approaches: sequential, pure MPI, pure OpenMP, hybrid MPI+OpenMP, and with computation/communication overlap.


## Target Architecture

**PlaFRIM Cluster**: 
- **Nodes**: 2 available nodes
- **Architecture per node**: 4 NUMA domains × 6 cores = 24 cores/node
- **Total**: 48 cores (2 nodes)

**Hybrid MPI+OpenMP Configuration**:
- 1 MPI process per NUMA domain (4 processes/node)
- 6 OpenMP threads per MPI process
- Testable process numbers: **multiples of 6** (6, 12, 18, 24, 30, 36, 42, 48)

---

## Compilation

### Standard compilation
```bash
make              # Compile all base versions
make clean        # Remove binaries
```

### Specific versions
```bash
make stencil                 # Sequential version
make stencil_mpi             # Pure MPI version
make stencil_openmp          # Pure OpenMP version
make stencil_mpi_openmp      # Hybrid MPI+OpenMP version
make stencil_recouvrement    # Version with computation/communication overlap
```

---

##  Available Versions

| Version | Code | Description |
|---------|------|-------------|
| **Sequential** | `seq` | Single-core execution |
| **MPI** | `mpi` | Pure distributed parallelization |
| **OpenMP** | `omp` | Pure shared-memory parallelization |
| **MPI+OpenMP** | `mpi_omp` | Hybrid: 1 MPI proc/NUMA, 6 threads/proc |
| **Overlap** | `rec` | Hybrid with computation/communication overlap |

---

## Benchmark Scripts

### 1. **Strong Scaling** (`benchmark_strong_scaling.py`)

Tests performance with a **fixed problem size** and increasing number of processes/threads.

#### Available options
```bash
-m, --modes      Versions to test (seq, mpi, omp, mpi_omp, rec)
-s, --size       Matrix size (default: 512)
-p, --max_p      Number(s) of processes/threads to test
-n, --nodes      Number of PlaFRIM nodes (default: 1)
-h, --help       Display help
```

#### Usage examples

**Multi-version test on 2 nodes**
```bash
python3 benchmark_strong_scaling.py -m mpi mpi_omp rec -s 512 -n 2
```
→ Tests MPI, MPI+OpenMP and overlap on a 512×512 grid with 2 nodes.

**MPI-only test on 1 node**
```bash
python3 benchmark_strong_scaling.py -m mpi -s 512 -n 1
```
→ Tests only MPI on 1 node.

**Test with custom sequence**
```bash
python3 benchmark_strong_scaling.py -m mpi_omp -s 1024 -p 6 12 24 48 -n 2
```
→ Tests MPI+OpenMP with 6, 12, 24 then 48 processes (1024×1024 grid).

**OpenMP test with different thread counts**
```bash
python3 benchmark_strong_scaling.py -m omp -s 512 -p 1 2 4 8 12 24
```
→ Tests OpenMP with 1, 2, 4, 8, 12 and 24 threads.

#### Constraints

- **`mpi_omp` and `rec` versions**: Process numbers **multiples of 6 only** (6, 12, 18, 24, 30, 36, 42, 48)
- **Multi-node**: Allocate nodes BEFORE with `salloc -N <number>`

---

### 2. **Weak Scaling** (`benchmark_weak_scaling.py`)

Tests performance with a **fixed workload per process/thread** and increasing number of processes/threads.

#### Available options
```bash
-m, --modes      Versions to test (seq, mpi, omp, mpi_omp, rec)
-s, --size       Workload per process/thread (default: 256)
-p, --max_p      Number(s) of processes/threads to test
-n, --nodes      Number of PlaFRIM nodes (default: 1)
-h, --help       Display help
```

#### Usage examples

**Multi-version test with workload 103**
```bash
python3 benchmark_weak_scaling.py -m mpi mpi_omp rec -s 103 -n 2
```
→ Tests MPI, MPI+OpenMP and overlap with load 103/process on 2 nodes.

**OpenMP test with custom sequence**
```bash
python3 benchmark_weak_scaling.py -m omp -s 103 -p 12 24
```
→ Tests OpenMP with 12 then 24 threads (load 103/thread).

**Complete hybrid test**
```bash
python3 benchmark_weak_scaling.py -m mpi_omp -s 103 -p 6 12 24 48 -n 2
```
→ Tests MPI+OpenMP with 6, 12, 24, 48 processes (load 103/process).

#### Recommendation

- Use `-s 103` instead of 256 for **faster tests**.

---

### 3. **Comparison** (`benchmark_comparaison.py`)

Compares all available versions using **full computing capacity** (all cores).

#### Available options
```bash
-s, --size       Matrix size (default: 512)
-h, --help       Display help
```

#### Usage examples

**Comparison with default size**
```bash
python3 benchmark_comparaison.py
```
→ Compares all versions on 512×512 grid with all available cores.

**Comparison with custom grid**
```bash
python3 benchmark_comparaison.py -s 1024
```
→ Compares all versions on 1024×1024 grid.


---

##  References

- **OpenMPI Documentation**: https://www.open-mpi.org/doc/
- **Intel MPI Documentation**: https://www.intel.com/content/www/us/en/docs/mpi-library/
- **PlaFRIM**: https://www.plafrim.fr/
