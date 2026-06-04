#!/usr/bin/env bash
#
# run_all.sh - Pipeline completo en UNA maquina: compila las tres
# versiones (seq, OpenMP, CUDA), genera los datasets que falten y corre el
# benchmark, dejando los tiempos crudos en un CSV. Pensado para una maquina
# con GPU NVIDIA por terminal/SSH.
#
# Correr todo en la misma maquina da una linea base secuencial comun,
# necesaria para que speedup(OMP) y speedup(CUDA) sean comparables.
#
# Uso:
#   bash scripts/run_all.sh
#
# Variables opcionales (heredadas por run_benchmark.sh):
#   REPS=5  MAX_ITER=100  K_LIST="3 5 10 200"  THREADS_LIST=auto  CSV=results/benchmark.csv

set -e
cd "$(dirname "$0")/.."

REPS=${REPS:-5}
MAX_ITER=${MAX_ITER:-100}
CSV=${CSV:-results/benchmark.csv}

echo "==================================================================="
echo " K-means: pipeline completo (seq + OpenMP + CUDA) en esta maquina"
echo "==================================================================="

# --- 1. Compilar ---------------------------------------------------------
echo ""
echo ">>> [1/4] Compilando..."
make clean
make cpu
if make cuda 2>/dev/null; then
    echo "    CUDA compilado: las tres versiones entraran al benchmark."
else
    echo "    AVISO: CUDA no compilo (¿sin nvcc/GPU?). Se mide solo seq + OMP."
fi
ls -la kmeans_seq kmeans_omp kmeans_cuda 2>/dev/null || true

# --- 2. Datasets ---------------------------------------------------------
echo ""
echo ">>> [2/4] Generando datasets que falten..."
mkdir -p data results
gen() { [ -f "$4" ] && echo "    ya existe $4" || python3 scripts/gen_dataset.py "$1" "$2" "$3" "$4"; }
gen 1000   2  3  data/small_2d.txt
gen 10000  10 5  data/medium_10d.txt
gen 100000 50 10 data/large_50d.txt
[ -f data/punto0_kmeans.txt ] || echo "    AVISO: falta data/punto0_kmeans.txt (dataset real); se omite."

# --- 3. Detectar nucleos y armar THREADS_LIST ----------------------------
echo ""
echo ">>> [3/4] Detectando nucleos..."
NCPU=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
if [ -z "${THREADS_LIST:-}" ]; then
    THREADS_LIST=$(python3 - "$NCPU" <<'PY'
import sys
n = int(sys.argv[1])
ths, t = [], 1
while t <= n:
    ths.append(t); t *= 2
if n not in ths: ths.append(n)
ths.append(min(n*2, n+8))
print(' '.join(map(str, sorted(set(ths)))))
PY
)
fi
echo "    nucleos=$NCPU  THREADS_LIST=[$THREADS_LIST]"

# --- 4. Benchmark -------------------------------------------------------
echo ""
echo ">>> [4/4] Benchmark (REPS=$REPS MAX_ITER=$MAX_ITER)..."
REPS="$REPS" MAX_ITER="$MAX_ITER" THREADS_LIST="$THREADS_LIST" CSV="$CSV" \
    bash scripts/run_benchmark.sh

echo ""
echo "==================================================================="
echo " Listo. Resultados crudos (tiempos por version/K/hebras) en:"
echo "   $CSV"
echo "==================================================================="
