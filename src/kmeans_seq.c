/*
 * kmeans_seq.c  -  Version SECUENCIAL de K-means (linea base)
 *
 * Compilar:
 *   gcc -O2 -o kmeans_seq kmeans_seq.c -lm
 * Ejecutar:
 *   ./kmeans_seq <dataset> <K> <max_iter> [archivo_labels]
 *
 * Esta version es la referencia contra la cual se mide el speedup
 * de las versiones OpenMP y CUDA.
 *
 * Proyecto Semestral - Introduccion a la Computacion Paralela
 */

#include "kmeans_common.h"

/* Reloj de pared portable. clock_gettime no existe en MSVC (Windows
 * nativo, el compilador que usa nvcc/Visual Studio): alli se usa
 * QueryPerformanceCounter. En Linux/macOS se mantiene clock_gettime. */
#ifdef _WIN32
#include <windows.h>
static double wall_time(void) {
    LARGE_INTEGER freq, t;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart;
}
#else
#include <time.h>
static double wall_time(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}
#endif

/* Distancia euclidiana al cuadrado entre punto p y centroide c.
 * Se usa el cuadrado porque para comparar/asignar no hace falta
 * la raiz, ahorrando calculo. */
static double dist2(const float *p, const float *c, int D) {
    double s = 0.0;
    for (int d = 0; d < D; d++) {
        double diff = (double)p[d] - (double)c[d];
        s += diff * diff;
    }
    return s;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Uso: %s <dataset> <K> <max_iter> [labels_out]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *dataset = argv[1];
    int K        = atoi(argv[2]);
    int max_iter = atoi(argv[3]);
    const char *labels_out = (argc > 4) ? argv[4] : NULL;

    int N, D;
    float *data = load_dataset(dataset, &N, &D);

    float *centroids = (float *)malloc((size_t)K * D * sizeof(float));
    float *new_sum   = (float *)malloc((size_t)K * D * sizeof(float));
    int   *count     = (int   *)malloc((size_t)K * sizeof(int));
    int   *labels    = (int   *)malloc((size_t)N * sizeof(int));

    init_centroids(data, N, D, K, centroids, 42u);

    double t0 = wall_time();

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        /* --- Paso de asignacion: cada punto -> centroide mas cercano --- */
        int changed = 0;
        for (int i = 0; i < N; i++) {
            const float *p = &data[(size_t)i * D];
            double best = 1e300;
            int best_c = 0;
            for (int c = 0; c < K; c++) {
                double dd = dist2(p, &centroids[(size_t)c * D], D);
                if (dd < best) { best = dd; best_c = c; }
            }
            if (labels[i] != best_c) { changed = 1; }
            labels[i] = best_c;
        }

        /* --- Paso de actualizacion: recomputar centroides --- */
        memset(new_sum, 0, (size_t)K * D * sizeof(float));
        memset(count,   0, (size_t)K * sizeof(int));
        for (int i = 0; i < N; i++) {
            int c = labels[i];
            const float *p = &data[(size_t)i * D];
            for (int d = 0; d < D; d++) new_sum[(size_t)c * D + d] += p[d];
            count[c]++;
        }
        for (int c = 0; c < K; c++) {
            if (count[c] > 0) {
                for (int d = 0; d < D; d++)
                    centroids[(size_t)c * D + d] = new_sum[(size_t)c * D + d] / count[c];
            }
        }

        if (!changed) { iter++; break; }  /* convergencia */
    }

    double elapsed = wall_time() - t0;

    printf("SEQ    N=%d D=%d K=%d iters=%d tiempo=%.4f s\n", N, D, K, iter, elapsed);

    if (labels_out) save_assignments(labels_out, labels, N);

    free(data); free(centroids); free(new_sum); free(count); free(labels);
    return EXIT_SUCCESS;
}
