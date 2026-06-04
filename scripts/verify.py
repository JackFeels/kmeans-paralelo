#!/usr/bin/env python3
"""
verify.py - Verifica que dos archivos de asignaciones (labels) sean
equivalentes salvo permutacion de etiquetas.

K-means puede numerar los clusters distinto entre versiones aunque el
agrupamiento sea identico. Por eso comparamos la PARTICION, no las
etiquetas exactas: dos asignaciones son equivalentes si inducen los
mismos grupos de puntos.

Uso:
    python3 verify.py labels_a.txt labels_b.txt
"""
import sys


def load(path):
    with open(path) as f:
        return [int(x) for x in f.read().split()]


def partition_signature(labels):
    """Agrupa indices de puntos por etiqueta y devuelve un conjunto
    canonico de grupos (frozenset de frozensets)."""
    groups = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(lab, set()).add(idx)
    return frozenset(frozenset(g) for g in groups.values())


def main():
    if len(sys.argv) != 3:
        print("Uso: python3 verify.py labels_a.txt labels_b.txt")
        sys.exit(1)

    a = load(sys.argv[1])
    b = load(sys.argv[2])

    if len(a) != len(b):
        print(f"DIFERENTE: distinto numero de puntos ({len(a)} vs {len(b)})")
        sys.exit(1)

    if partition_signature(a) == partition_signature(b):
        print("OK: las particiones son identicas (clustering equivalente)")
    else:
        # Tolerancia: porcentaje de puntos en el mismo grupo
        same = sum(1 for x, y in zip(a, b) if x == y)
        print(f"Las particiones difieren. Coincidencia directa: "
              f"{100.0 * same / len(a):.2f}% "
              f"(puede deberse a renumeracion de clusters)")


if __name__ == "__main__":
    main()
