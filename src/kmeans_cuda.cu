/*
 * kmeans_cuda.cu  -  Version PARALELA en GPU con CUDA
 *
 * Compilar:
 *   nvcc -O2 -o kmeans_cuda kmeans_cuda.cu
 * Ejecutar:
 *   ./kmeans_cuda <dataset> <K> <max_iter> [labels_out]
 *
 * Estrategia de paralelizacion (paralelismo masivo en GPU):
 *  - Paso de asignacion: se lanza un hilo por punto. Cada hilo calcula
 *    la distancia de "su" punto a los K centroides y elige el minimo.
 *    Los centroides se copian a memoria compartida del bloque para
 *    acelerar los accesos (son leidos por todos los hilos del bloque).
 *  - Paso de actualizacion: se acumulan sumas y conteos por cluster
 *    usando atomicAdd. Para K*D moderado esto es eficiente; si crece
 *    mucho, conviene una reduccion por bloques (mencionado en el informe).
 *  - Los centroides se dividen por el conteo en un kernel final.
 *
 * Nota: requiere una GPU NVIDIA. Si no tienes una localmente, puedes
 * compilar y ejecutar gratis en Google Colab (Runtime -> GPU).
 *
 * Proyecto Semestral - Introduccion a la Computacion Paralela
 */

#include "kmeans_common.h"
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256

/* Macro para chequear errores de CUDA */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* ---------------- Kernel 1: asignacion ---------------- */
__global__ void assign_kernel(const float *data, const float *centroids,
                              int *labels, int N, int D, int K) {
    extern __shared__ float s_centroids[];  /* K*D floats */

    /* Carga cooperativa de centroides a memoria compartida */
    for (int idx = threadIdx.x; idx < K * D; idx += blockDim.x)
        s_centroids[idx] = centroids[idx];
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float *p = &data[(size_t)i * D];
    float best = 3.4e38f;
    int best_c = 0;
    for (int c = 0; c < K; c++) {
        float s = 0.0f;
        const float *cc = &s_centroids[c * D];
        for (int d = 0; d < D; d++) {
            float diff = p[d] - cc[d];
            s += diff * diff;
        }
        if (s < best) { best = s; best_c = c; }
    }
    labels[i] = best_c;
}

/* ---------------- Kernel 2: acumulacion ---------------- */
__global__ void accumulate_kernel(const float *data, const int *labels,
                                  float *sums, int *counts, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int c = labels[i];
    const float *p = &data[(size_t)i * D];
    for (int d = 0; d < D; d++)
        atomicAdd(&sums[(size_t)c * D + d], p[d]);
    atomicAdd(&counts[c], 1);
}

/* ---------------- Kernel 3: division ---------------- */
__global__ void update_kernel(float *centroids, const float *sums,
                              const int *counts, int K, int D) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= K) return;
    int n = counts[c];
    if (n > 0) {
        for (int d = 0; d < D; d++)
            centroids[(size_t)c * D + d] = sums[(size_t)c * D + d] / n;
    }
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

    float *h_centroids = (float *)malloc((size_t)K * D * sizeof(float));
    int   *h_labels    = (int   *)malloc((size_t)N * sizeof(int));
    init_centroids(data, N, D, K, h_centroids, 42u);

    /* Reserva en GPU */
    float *d_data, *d_centroids, *d_sums;
    int   *d_labels, *d_counts;
    CUDA_CHECK(cudaMalloc(&d_data,      (size_t)N * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_centroids, (size_t)K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sums,      (size_t)K * D * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_labels,    (size_t)N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_counts,    (size_t)K * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_data, data, (size_t)N * D * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids, (size_t)K * D * sizeof(float),
                          cudaMemcpyHostToDevice));

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    size_t shmem = (size_t)K * D * sizeof(float);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    int iter;
    for (iter = 0; iter < max_iter; iter++) {
        assign_kernel<<<blocks, THREADS_PER_BLOCK, shmem>>>(
            d_data, d_centroids, d_labels, N, D, K);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemset(d_sums,   0, (size_t)K * D * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_counts, 0, (size_t)K * sizeof(int)));

        accumulate_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_data, d_labels, d_sums, d_counts, N, D);
        CUDA_CHECK(cudaGetLastError());

        int kblocks = (K + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        update_kernel<<<kblocks, THREADS_PER_BLOCK>>>(
            d_centroids, d_sums, d_counts, K, D);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));

    printf("CUDA   N=%d D=%d K=%d iters=%d tiempo=%.4f s\n",
           N, D, K, iter, ms / 1000.0f);

    if (labels_out) {
        CUDA_CHECK(cudaMemcpy(h_labels, d_labels, (size_t)N * sizeof(int),
                              cudaMemcpyDeviceToHost));
        save_assignments(labels_out, h_labels, N);
    }

    cudaFree(d_data); cudaFree(d_centroids); cudaFree(d_sums);
    cudaFree(d_labels); cudaFree(d_counts);
    free(data); free(h_centroids); free(h_labels);
    return EXIT_SUCCESS;
}
