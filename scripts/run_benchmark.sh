#!/usr/bin/env bash
#
# run_benchmark.sh - Ejecuta las tres versiones de K-means sobre los
# datasets registrando tiempos. Sweep de K y numero de hebras, con
# repeticiones. El CSV resultante es "crudo": una fila por repeticion
# (version, dataset, N, D, K, threads, rep, iters, tiempo_s).
#
# Uso:
#   bash scripts/run_benchmark.sh
#
# Variables de entorno opcionales:
#   REPS=5                       repeticiones por configuracion
#   MAX_ITER=50                  max iteraciones de K-means
#   K_LIST="3 5 10 200"          valores de K a barrer (200 ~ caso real FFClust)
#   THREADS_LIST="1 2 4 8 10"    numero de hebras para OMP
#   CSV=results/benchmark.csv    archivo de salida
#
# Genera (por defecto): results/benchmark.csv con schema
#   version,dataset,N,D,K,threads,rep,iters,tiempo_s

set -e
cd "$(dirname "$0")/.."

REPS=${REPS:-5}
MAX_ITER=${MAX_ITER:-50}
K_LIST=${K_LIST:-"3 5 10 200"}
THREADS_LIST=${THREADS_LIST:-"1 2 4 8 10"}
CSV=${CSV:-results/benchmark.csv}

DATASETS=(
    "data/small_2d.txt"
    "data/medium_10d.txt"
    "data/large_50d.txt"
    "data/punto0_kmeans.txt"   # dataset real (coords 3D, ~955k puntos)
)

mkdir -p "$(dirname "$CSV")"
echo "version,dataset,N,D,K,threads,rep,iters,tiempo_s" > "$CSV"

# Extrae el valor de un campo "key=valor" en una linea (key puede repetir
# solo una vez por linea, que es nuestro caso).
extract_field() {
    local key=$1
    local line=$2
    echo "$line" | sed -n "s/.*$key=\([0-9.][0-9.]*\).*/\1/p"
}

run_one() {
    local version=$1 ds=$2 k=$3 th=$4 rep=$5
    local out
    case "$version" in
        seq)  out=$(./kmeans_seq  "$ds" "$k" "$MAX_ITER") ;;
        omp)  out=$(OMP_NUM_THREADS=$th ./kmeans_omp "$ds" "$k" "$MAX_ITER") ;;
        cuda) out=$(./kmeans_cuda "$ds" "$k" "$MAX_ITER") ;;
    esac
    local N D iters t
    N=$(extract_field "N"      "$out")
    D=$(extract_field "D"      "$out")
    iters=$(extract_field "iters" "$out")
    t=$(extract_field "tiempo" "$out")
    echo "$version,$ds,$N,$D,$k,$th,$rep,$iters,$t" >> "$CSV"
    printf "  %-4s K=%-2s threads=%-2s rep=%s -> %s s\n" "$version" "$k" "$th" "$rep" "$t"
}

echo "Benchmark: REPS=$REPS MAX_ITER=$MAX_ITER K_LIST=[$K_LIST] THREADS_LIST=[$THREADS_LIST]"
echo ""

for ds in "${DATASETS[@]}"; do
    if [ ! -f "$ds" ]; then
        echo "AVISO: falta $ds (genera los datasets primero); saltando."
        continue
    fi
    echo "=== Dataset: $ds ==="
    for k in $K_LIST; do
        echo "-- K=$k --"
        # Secuencial (linea base) -> threads=1 por convencion
        for rep in $(seq 1 "$REPS"); do
            run_one seq "$ds" "$k" 1 "$rep"
        done
        # OpenMP variando hebras
        for th in $THREADS_LIST; do
            for rep in $(seq 1 "$REPS"); do
                run_one omp "$ds" "$k" "$th" "$rep"
            done
        done
        # CUDA solo si el binario existe (compilado en Colab o GPU local)
        if [ -x ./kmeans_cuda ]; then
            for rep in $(seq 1 "$REPS"); do
                run_one cuda "$ds" "$k" 0 "$rep"
            done
        fi
    done
done

echo ""
echo "Resultados crudos en $CSV"
echo "Filas: $(($(wc -l < "$CSV") - 1)) (sin cabecera)"
