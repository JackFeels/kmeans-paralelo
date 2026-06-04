#!/usr/bin/env python3
"""
gen_dataset.py - Genera datasets sinteticos con clusters gaussianos.

Crea puntos agrupados alrededor de K centros aleatorios, lo que da
una estructura de clusters real para que K-means tenga algo que
encontrar (a diferencia de ruido uniforme).

Formato de salida (texto plano):
    N D
    x0 x1 ... x(D-1)     <- una linea por punto

Uso:
    python3 gen_dataset.py <N> <D> <K> <salida> [semilla]

Ejemplos (segun el diseno experimental de la propuesta):
    python3 gen_dataset.py 1000   2  3  data/small_2d.txt
    python3 gen_dataset.py 10000  10 5  data/medium_10d.txt
    python3 gen_dataset.py 100000 50 10 data/large_50d.txt
"""
import sys
import numpy as np


def main():
    if len(sys.argv) < 5:
        print("Uso: python3 gen_dataset.py <N> <D> <K> <salida> [semilla]")
        sys.exit(1)

    N = int(sys.argv[1])
    D = int(sys.argv[2])
    K = int(sys.argv[3])
    out = sys.argv[4]
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 7

    rng = np.random.default_rng(seed)

    # Centros de los clusters distribuidos en el espacio
    centers = rng.uniform(-50.0, 50.0, size=(K, D))

    # Reparto (aprox) equitativo de puntos entre clusters
    counts = [N // K] * K
    for i in range(N - sum(counts)):
        counts[i] += 1

    parts = []
    for c in range(K):
        # Dispersion gaussiana alrededor de cada centro
        pts = rng.normal(loc=centers[c], scale=5.0, size=(counts[c], D))
        parts.append(pts)

    data = np.vstack(parts).astype(np.float32)
    rng.shuffle(data)  # mezclar para que el orden no revele el cluster

    # Escritura eficiente
    with open(out, "w") as f:
        f.write(f"{N} {D}\n")
        np.savetxt(f, data, fmt="%.6f")

    print(f"Generado {out}: N={N} D={D} K={K} (semilla={seed})")


if __name__ == "__main__":
    main()
