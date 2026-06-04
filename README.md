# K-means paralelo: secuencial, OpenMP y CUDA

Implementación de tres versiones del algoritmo de clustering **K-means**:

- **Secuencial** en C — línea base de referencia.
- **OpenMP** — paralela en CPU (memoria compartida).
- **CUDA** — paralela en GPU (NVIDIA).

Las tres comparten la carga de datos y la inicialización de centroides
(método de Forgy con semilla fija), de modo que producen el mismo
agrupamiento y los tiempos son comparables entre sí.

## Estructura

```
kmeans-paralelo/
├── Makefile                 detecta macOS (clang+libomp) / Linux (gcc, nvcc)
├── README.md
├── src/
│   ├── kmeans_common.h      utilidades compartidas (carga de datos, init)
│   ├── kmeans_seq.c         versión secuencial
│   ├── kmeans_omp.c         versión OpenMP (CPU)
│   └── kmeans_cuda.cu       versión CUDA (GPU)
├── scripts/
│   ├── gen_dataset.py       genera datasets sintéticos (clusters gaussianos)
│   ├── run_benchmark.sh     barrido de versiones/K/hebras -> CSV de tiempos
│   ├── analyze.py           tablas (CSV + Markdown) y figuras desde el CSV
│   ├── run_all.sh           compila + datos + benchmark + análisis (Linux/macOS)
│   ├── run_all.ps1          idem para Windows (Visual Studio + CUDA)
│   ├── kmeans_colab.ipynb   corre las 3 versiones en Google Colab (GPU)
│   └── verify.py            verifica equivalencia del clustering entre versiones
└── data/                    datasets de prueba
```

## Requisitos

- **Secuencial / OpenMP:** compilador C con soporte OpenMP.
  - Linux: `gcc` (incluye OpenMP).
  - macOS: Apple Clang + `libomp` (`brew install libomp`). El Makefile lo
    detecta automáticamente.
- **CUDA:** CUDA Toolkit (`nvcc`) y una GPU NVIDIA.
- **Datasets:** Python 3 con `numpy` (solo para `gen_dataset.py`).

## Compilación

```bash
make            # compila kmeans_seq y kmeans_omp (no requiere GPU)
make cuda       # compila kmeans_cuda (requiere nvcc + GPU NVIDIA)
make all        # las tres
make clean
```

## Datasets

Formato de archivo: primera línea `N D` (número de puntos y dimensiones),
luego N líneas con D valores cada una.

```bash
python3 scripts/gen_dataset.py 1000   2  3  data/small_2d.txt
python3 scripts/gen_dataset.py 10000  10 5  data/medium_10d.txt
python3 scripts/gen_dataset.py 100000 50 10 data/large_50d.txt
```

El repositorio incluye `data/punto0_kmeans.txt`, un dataset real de
~955.000 puntos 3D.

## Ejecución

```bash
# Secuencial
./kmeans_seq data/punto0_kmeans.txt 5 100

# OpenMP (variando el número de hebras)
OMP_NUM_THREADS=4 ./kmeans_omp data/punto0_kmeans.txt 5 100

# CUDA
./kmeans_cuda data/punto0_kmeans.txt 5 100
```

Argumentos: `<dataset> <K> <max_iter> [archivo_labels_salida]`
Cada ejecución imprime N, D, K, iteraciones realizadas y tiempo en segundos.

## Verificación de correctitud

Confirma que dos versiones producen el mismo agrupamiento:

```bash
./kmeans_seq data/punto0_kmeans.txt 5 100 seq.txt
OMP_NUM_THREADS=8 ./kmeans_omp data/punto0_kmeans.txt 5 100 omp.txt
python3 scripts/verify.py seq.txt omp.txt
```

En datasets grandes pueden aparecer diferencias mínimas (<0,01% de puntos)
por el orden de las sumas en punto flotante al paralelizar; el clustering
es equivalente.

## Benchmark

Las tres versiones en una misma máquina (línea base secuencial común):

```bash
# Linux / macOS (CUDA se omite si no hay nvcc)
bash scripts/run_all.sh

# Windows (Visual Studio + CUDA)
powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
```

Genera `results/benchmark.csv` con los tiempos por versión, K y número de
hebras. Variables opcionales: `REPS`, `MAX_ITER`, `K_LIST`, `THREADS_LIST`.

### Análisis (tablas y figuras)

`run_all.sh` / `run_all.ps1` llaman automáticamente a `analyze.py`, que
también puede ejecutarse por separado sobre cualquier CSV de tiempos:

```bash
python3 scripts/analyze.py results/benchmark.csv
# Combinar tiempos de varias máquinas (p.ej. CPU + GPU):
python3 scripts/analyze.py cpu.csv gpu.csv
```

Produce, para cualquier combinación de versiones presente en el CSV:
- `results/tables/summary.csv` y `summary.md` — mediana, speedup y eficiencia.
- `results/figures/speedup_*.png`, `efficiency_*.png`, `comparison_*.png`.

Requiere Python con `matplotlib`.

### CUDA sin GPU local: Google Colab

`scripts/kmeans_colab.ipynb` clona este repositorio, compila las tres
versiones y corre el benchmark en una GPU de Colab (Runtime → GPU). No
requiere instalar nada localmente.
