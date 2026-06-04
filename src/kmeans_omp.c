/*
 * kmeans_omp.c  -  Version PARALELA en CPU con OpenMP
 *
 * Compilar:
 *   gcc -O2 -fopenmp -o kmeans_omp kmeans_omp.c -lm
 * Ejecutar:
 *   OMP_NUM_THREADS=4 ./kmeans_omp <dataset> <K> <max_iter> [labels_out]
 *
 * Estrategia de paralelizacion (paralelismo de datos):
 *  - Paso de asignacion: el bucle sobre los N puntos se reparte entre
 *    hebras con #pragma omp parallel for. Cada punto es independiente,
 *    por lo que no hay condiciones de carrera al escribir labels[i].
 *  - Paso de actualizacion: usamos acumuladores locales por hebra para
 *    evitar contencion sobre los centroides, y luego una reduccion
 *    manual en region critica. Esto reduce el overhead de sincronizacion.
 *
 * Proyecto Semestral - Introduccion a la Computacion Paralela
 */

#include "kmeans_common.h"
#include <omp.h>

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

    int nthreads = omp_get_max_threads();

    double t0 = omp_get_wtime();

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        int changed = 0;
        int i;  /* declarada fuera del for: OpenMP 2.0 (MSVC) lo exige */

        /* --- Paso de asignacion paralelo --- */
        #pragma omp parallel for schedule(static) reduction(|:changed)
        for (i = 0; i < N; i++) {
            const float *p = &data[(size_t)i * D];
            double best = 1e300;
            int best_c = 0;
            for (int c = 0; c < K; c++) {
                double dd = dist2(p, &centroids[(size_t)c * D], D);
                if (dd < best) { best = dd; best_c = c; }
            }
            if (labels[i] != best_c) changed = 1;
            labels[i] = best_c;
        }

        /* --- Paso de actualizacion con acumuladores por hebra --- */
        memset(new_sum, 0, (size_t)K * D * sizeof(float));
        memset(count,   0, (size_t)K * sizeof(int));

        #pragma omp parallel
        {
            /* Acumuladores privados de cada hebra */
            float *local_sum = (float *)calloc((size_t)K * D, sizeof(float));
            int   *local_cnt = (int   *)calloc((size_t)K,     sizeof(int));

            #pragma omp for schedule(static) nowait
            for (i = 0; i < N; i++) {
                int c = labels[i];
                const float *p = &data[(size_t)i * D];
                for (int d = 0; d < D; d++) local_sum[(size_t)c * D + d] += p[d];
                local_cnt[c]++;
            }

            /* Reduccion manual hacia los acumuladores globales */
            #pragma omp critical
            {
                for (int c = 0; c < K; c++) {
                    count[c] += local_cnt[c];
                    for (int d = 0; d < D; d++)
                        new_sum[(size_t)c * D + d] += local_sum[(size_t)c * D + d];
                }
            }
            free(local_sum);
            free(local_cnt);
        }

        for (int c = 0; c < K; c++) {
            if (count[c] > 0) {
                for (int d = 0; d < D; d++)
                    centroids[(size_t)c * D + d] = new_sum[(size_t)c * D + d] / count[c];
            }
        }

        if (!changed) { iter++; break; }
    }

    double elapsed = omp_get_wtime() - t0;

    printf("OMP    N=%d D=%d K=%d iters=%d hebras=%d tiempo=%.4f s\n",
           N, D, K, iter, nthreads, elapsed);

    if (labels_out) save_assignments(labels_out, labels, N);

    free(data); free(centroids); free(new_sum); free(count); free(labels);
    return EXIT_SUCCESS;
}
