#ifndef KMEANS_COMMON_H
#define KMEANS_COMMON_H

/*
 * kmeans_common.h
 * Utilidades compartidas por las tres versiones de K-means
 * (secuencial, OpenMP, CUDA).
 *
 * Formato del dataset (texto plano):
 *   primera linea:  N D            (numero de puntos, dimensiones)
 *   siguientes N:   x0 x1 ... x(D-1)
 *
 * Proyecto Semestral - Introduccion a la Computacion Paralela
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ----------------------------------------------------------------
 * Carga de dataset desde archivo de texto.
 * Reserva memoria para data (N*D floats en layout row-major).
 * El llamador es responsable de liberar *data con free().
 * ---------------------------------------------------------------- */
static inline float *load_dataset(const char *path, int *N_out, int *D_out) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Error: no se pudo abrir el dataset '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    int N, D;
    if (fscanf(f, "%d %d", &N, &D) != 2) {
        fprintf(stderr, "Error: cabecera invalida en '%s'\n", path);
        exit(EXIT_FAILURE);
    }
    float *data = (float *)malloc((size_t)N * D * sizeof(float));
    if (!data) {
        fprintf(stderr, "Error: sin memoria para %d x %d floats\n", N, D);
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < (size_t)N * D; i++) {
        if (fscanf(f, "%f", &data[i]) != 1) {
            fprintf(stderr, "Error: datos insuficientes en '%s'\n", path);
            exit(EXIT_FAILURE);
        }
    }
    fclose(f);
    *N_out = N;
    *D_out = D;
    return data;
}

/* ----------------------------------------------------------------
 * Inicializacion de centroides: metodo Forgy con semilla fija.
 * Elige K puntos del dataset como centroides iniciales.
 * La semilla fija garantiza que todas las versiones partan igual,
 * de modo que el speedup se mida de forma justa.
 * ---------------------------------------------------------------- */
static inline void init_centroids(const float *data, int N, int D, int K,
                                  float *centroids, unsigned int seed) {
    srand(seed);
    for (int c = 0; c < K; c++) {
        int idx = rand() % N;
        for (int d = 0; d < D; d++) {
            centroids[(size_t)c * D + d] = data[(size_t)idx * D + d];
        }
    }
}

/* ----------------------------------------------------------------
 * Guarda las asignaciones finales (cluster por punto) en archivo.
 * Util para verificar que las tres versiones dan el mismo resultado.
 * ---------------------------------------------------------------- */
static inline void save_assignments(const char *path, const int *labels, int N) {
    FILE *f = fopen(path, "w");
    if (!f) return;
    for (int i = 0; i < N; i++) fprintf(f, "%d\n", labels[i]);
    fclose(f);
}

#endif /* KMEANS_COMMON_H */
