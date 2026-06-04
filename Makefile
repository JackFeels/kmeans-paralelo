# Makefile - Proyecto Semestral K-means paralelo
#
#   make            -> compila seq y omp (no requiere GPU)
#   make cuda       -> compila la version CUDA (requiere nvcc/GPU)
#   make all        -> intenta compilar las tres
#   make clean      -> elimina binarios
#
# Detecta automaticamente la plataforma:
#   macOS  -> Apple Clang + libomp de Homebrew (libomp no es Apple Clang
#             nativo: requiere -Xpreprocessor -fopenmp y -lomp).
#   Linux  -> gcc -fopenmp directamente.

UNAME_S := $(shell uname -s)

CFLAGS  = -O2 -Wall
LDLIBS  = -lm
NVCC    = nvcc

ifeq ($(UNAME_S),Darwin)
    CC          = clang
    # Resolver el prefix de libomp via Homebrew; si falla usar la ruta
    # canonica en Apple Silicon. El usuario debe tener `brew install libomp`.
    BREW_LIBOMP := $(shell brew --prefix libomp 2>/dev/null)
    ifeq ($(BREW_LIBOMP),)
        BREW_LIBOMP := /opt/homebrew/opt/libomp
    endif
    OMP_CFLAGS  = -Xpreprocessor -fopenmp -I$(BREW_LIBOMP)/include
    OMP_LDLIBS  = -L$(BREW_LIBOMP)/lib -lomp
else
    CC          = gcc
    OMP_CFLAGS  = -fopenmp
    OMP_LDLIBS  =
endif

SRC = src

.PHONY: cpu cuda all clean

cpu: kmeans_seq kmeans_omp

kmeans_seq: $(SRC)/kmeans_seq.c $(SRC)/kmeans_common.h
	$(CC) $(CFLAGS) -o $@ $(SRC)/kmeans_seq.c $(LDLIBS)

kmeans_omp: $(SRC)/kmeans_omp.c $(SRC)/kmeans_common.h
	$(CC) $(CFLAGS) $(OMP_CFLAGS) -o $@ $(SRC)/kmeans_omp.c $(LDLIBS) $(OMP_LDLIBS)

# Requiere CUDA toolkit instalado
cuda: kmeans_cuda

kmeans_cuda: $(SRC)/kmeans_cuda.cu $(SRC)/kmeans_common.h
	$(NVCC) -O2 -o $@ $(SRC)/kmeans_cuda.cu

all: cpu cuda

clean:
	rm -f kmeans_seq kmeans_omp kmeans_cuda
