#!/usr/bin/env python3
"""
analyze.py - Genera resultados y analisis a partir del CSV de tiempos.

UN SOLO script que funciona para cualquier combinacion de versiones
presente en el CSV (solo seq, seq+omp, o seq+omp+cuda). Produce:

    results/tables/summary.csv   tabla: mediana, min, max, speedup, eficiencia
    results/tables/summary.md    misma tabla en Markdown (pegar en el informe)
    results/figures/speedup_<dataset>.png
    results/figures/efficiency_<dataset>.png
    results/figures/comparison_<dataset>.png   (seq vs OMP-best vs CUDA)

Speedup y eficiencia se calculan respecto a la version SECUENCIAL del
MISMO dataset y MISMO K (linea base):
    S(p) = T_seq / T_p      E(p) = S(p) / p

Uso:
    python3 scripts/analyze.py [csv1 csv2 ...]

Sin argumentos usa results/benchmark.csv. Acepta varios CSV para combinar
(p.ej. tiempos de CPU de una maquina + tiempos de GPU de otra).
"""
import csv
import os
import sys
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")          # backend sin pantalla
import matplotlib.pyplot as plt


# ---------------------------------------------------------------- carga
def load_rows(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            r["N"]        = int(r["N"])
            r["D"]        = int(r["D"])
            r["K"]        = int(r["K"])
            r["threads"]  = int(r["threads"])
            r["iters"]    = int(r["iters"])
            r["tiempo_s"] = float(r["tiempo_s"])
            r["dataset"]  = os.path.basename(r["dataset"]).replace(".txt", "")
            rows.append(r)
    return rows


def aggregate(rows):
    """(version, dataset, K, threads) -> {median, min, max, n}."""
    g = defaultdict(list)
    for r in rows:
        g[(r["version"], r["dataset"], r["K"], r["threads"])].append(r["tiempo_s"])
    return {k: {"median": statistics.median(v), "min": min(v),
                "max": max(v), "n": len(v)} for k, v in g.items()}


def datasets_and_ks(summary):
    ds = sorted({d for (_v, d, _k, _t) in summary})
    ks = sorted({k for (_v, _d, k, _t) in summary})
    return ds, ks


# ---------------------------------------------------------------- tablas
def _table_rows(summary):
    """Filas ordenadas con speedup y eficiencia calculados."""
    seq = {(d, k): s["median"]
           for (v, d, k, t), s in summary.items() if v == "seq"}
    out = []
    for (v, d, k, t) in sorted(summary):
        s = summary[(v, d, k, t)]
        base = seq.get((d, k))
        sp = (base / s["median"]) if (base and s["median"] > 0) else None
        eff = (sp / t) if (v == "omp" and t > 0 and sp is not None) else None
        out.append((v, d, k, t, s, sp, eff))
    return out


def write_summary_csv(summary, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["version", "dataset", "K", "threads", "n",
                    "median_s", "min_s", "max_s", "speedup", "efficiency"])
        for v, d, k, t, s, sp, eff in _table_rows(summary):
            w.writerow([v, d, k, t, s["n"],
                        f"{s['median']:.6f}", f"{s['min']:.6f}", f"{s['max']:.6f}",
                        f"{sp:.3f}" if sp is not None else "",
                        f"{eff:.3f}" if eff is not None else ""])


def write_summary_md(summary, path):
    with open(path, "w") as f:
        f.write("# Resumen de resultados\n\n")
        f.write("| Versión | Dataset | K | Hebras | Mediana (s) | Speedup | Eficiencia |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|\n")
        for v, d, k, t, s, sp, eff in _table_rows(summary):
            f.write(f"| {v} | {d} | {k} | {t} | {s['median']:.4f} | "
                    f"{sp:.2f}× | {eff*100:.0f}% |\n" if sp is not None and eff is not None
                    else f"| {v} | {d} | {k} | {t} | {s['median']:.4f} | "
                         f"{(f'{sp:.2f}×') if sp is not None else '—'} | — |\n")


# ---------------------------------------------------------------- figuras
def plot_speedup(summary, out_dir):
    ds_list, ks = datasets_and_ks(summary)
    for d in ds_list:
        plt.figure(figsize=(7, 5))
        drew = False
        for k in ks:
            base = summary.get(("seq", d, k, 1), {}).get("median")
            if base is None:
                continue
            ts = sorted(t for (v, dd, kk, t) in summary
                        if v == "omp" and dd == d and kk == k)
            xs, ys = [], []
            for t in ts:
                m = summary[("omp", d, k, t)]["median"]
                if m > 0:
                    xs.append(t); ys.append(base / m)
            if xs:
                plt.plot(xs, ys, "o-", label=f"OMP K={k}"); drew = True
        if not drew:
            plt.close(); continue
        all_t = sorted(t for (v, dd, _k, t) in summary if v == "omp" and dd == d)
        if all_t:
            plt.plot(all_t, all_t, "k--", alpha=0.4, label="ideal (S=p)")
        plt.title(f"Speedup vs hebras — {d}")
        plt.xlabel("Número de hebras (p)")
        plt.ylabel("Speedup  S(p) = T_seq / T_p")
        plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"speedup_{d}.png"), dpi=130)
        plt.close()


def plot_efficiency(summary, out_dir):
    ds_list, ks = datasets_and_ks(summary)
    for d in ds_list:
        plt.figure(figsize=(7, 5))
        drew = False
        for k in ks:
            base = summary.get(("seq", d, k, 1), {}).get("median")
            if base is None:
                continue
            ts = sorted(t for (v, dd, kk, t) in summary
                        if v == "omp" and dd == d and kk == k)
            xs, ys = [], []
            for t in ts:
                m = summary[("omp", d, k, t)]["median"]
                if m > 0 and t > 0:
                    xs.append(t); ys.append((base / m) / t)
            if xs:
                plt.plot(xs, ys, "o-", label=f"OMP K={k}"); drew = True
        if not drew:
            plt.close(); continue
        plt.axhline(1.0, color="k", ls="--", alpha=0.4, label="ideal (E=1)")
        plt.title(f"Eficiencia vs hebras — {d}")
        plt.xlabel("Número de hebras (p)")
        plt.ylabel("Eficiencia  E(p) = S(p) / p")
        plt.ylim(0, 1.2); plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"efficiency_{d}.png"), dpi=130)
        plt.close()


def plot_comparison(summary, out_dir):
    """Una figura por dataset: barras seq vs OMP-best (vs CUDA si existe)."""
    ds_list, ks = datasets_and_ks(summary)
    has_cuda = any(v == "cuda" for (v, _d, _k, _t) in summary)
    for d in ds_list:
        rows = []
        for k in ks:
            base = summary.get(("seq", d, k, 1), {}).get("median")
            if base is None:
                continue
            omp = [summary[("omp", d, k, t)]["median"]
                   for (v, dd, kk, t) in summary
                   if v == "omp" and dd == d and kk == k]
            omp_best = min(omp) if omp else None
            cuda = summary.get(("cuda", d, k, 0), {}).get("median")
            rows.append((k, base, omp_best, cuda))
        if not rows:
            continue
        x = list(range(len(rows)))
        plt.figure(figsize=(7, 5))
        if has_cuda:
            w = 0.25
            plt.bar([i - w for i in x], [r[1] for r in rows], w, label="Secuencial")
            plt.bar(x,                  [r[2] or 0 for r in rows], w, label="OpenMP (mejor)")
            plt.bar([i + w for i in x], [r[3] or 0 for r in rows], w, label="CUDA")
        else:
            w = 0.38
            plt.bar([i - w/2 for i in x], [r[1] for r in rows], w, label="Secuencial")
            plt.bar([i + w/2 for i in x], [r[2] or 0 for r in rows], w, label="OpenMP (mejor)")
        plt.xticks(x, [f"K={r[0]}" for r in rows])
        plt.ylabel("Tiempo (s)"); plt.yscale("log")
        plt.title(f"Tiempo de ejecución — {d}")
        plt.legend(); plt.grid(True, alpha=0.3, which="both"); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"comparison_{d}.png"), dpi=130)
        plt.close()


# ---------------------------------------------------------------- consola
def print_summary(summary):
    seq = {(d, k): s["median"]
           for (v, d, k, t), s in summary.items() if v == "seq"}
    hdr = f'{"ver":<5}{"dataset":<16}{"K":>4}{"thr":>5}{"med_s":>11}{"speedup":>9}{"effic":>7}'
    print(hdr); print("-" * len(hdr))
    for v, d, k, t, s, sp, eff in _table_rows(summary):
        sp_s = f"{sp:.2f}x" if (sp is not None and v != "seq") else "-"
        eff_s = f"{eff*100:.0f}%" if eff is not None else "-"
        print(f'{v:<5}{d:<16}{k:>4}{t:>5}{s["median"]:>11.4f}{sp_s:>9}{eff_s:>7}')


def main():
    csv_paths = sys.argv[1:] if len(sys.argv) > 1 else ["results/benchmark.csv"]
    tables_dir, figs_dir = "results/tables", "results/figures"
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)

    rows = []
    for p in csv_paths:
        if not os.path.exists(p):
            print(f"aviso: no existe {p}, se omite")
            continue
        rs = load_rows(p)
        print(f"  + {p}: {len(rs)} filas")
        rows.extend(rs)
    if not rows:
        print("error: no se cargo ningun CSV", file=sys.stderr)
        sys.exit(1)

    summary = aggregate(rows)
    write_summary_csv(summary, os.path.join(tables_dir, "summary.csv"))
    write_summary_md(summary, os.path.join(tables_dir, "summary.md"))
    plot_speedup(summary, figs_dir)
    plot_efficiency(summary, figs_dir)
    plot_comparison(summary, figs_dir)

    print(f"\nTablas:  {tables_dir}/summary.csv  y  summary.md")
    print(f"Figuras: {figs_dir}/*.png\n")
    print_summary(summary)


if __name__ == "__main__":
    main()
