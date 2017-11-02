/*
 * Copyright 2011-2015 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print a trace of CUDA API and GPU activity
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <stdio.h>
#include <string.h>

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define N_ELEMENTS 1048576
#define N_THREADS 1024
#define THREADS_PER_BLOCK 256
#define BLOCKS_PER_GRID ((N_THREADS + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

#define USE_CUPTI 1

extern void initTrace(void);
extern void finiTrace(void);

__global__ void VecAdd(const int* A, const int* B, int* C, int N)
{
  int num_elems = (N + (gridDim.x * blockDim.x) - 1) / (gridDim.x * blockDim.x);
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = num_elems * tid, j = 0; i < N && j < num_elems; i++, j++) {
    C[i] = A[i] + B[i];
  }
}

static void doPass(int* h_A, int* h_B, int* h_C, int* d_A, int* d_B, int* d_C,
                   size_t size, cudaStream_t stream) {
  RUNTIME_API_CALL(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream));
  RUNTIME_API_CALL(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream));

  VecAdd<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, 0, stream>>>(d_A, d_B, d_C, N_ELEMENTS);

  RUNTIME_API_CALL(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream));
}

int main(int argc, char **argv) {
  CUdevice device;
  char deviceName[32];
  int deviceNum = 0, devCount = 0;
  int **h_A1, **h_B1, **h_C1;
  int **h_A2, **h_B2, **h_C2;
  int **d_A1, **d_B1, **d_C1;
  int **d_A2, **d_B2, **d_C2;

  // initialize the activity trace
  // make sure activity is enabled before any CUDA API
#if USE_CUPTI
  initTrace();
  DRIVER_API_CALL(cuInit(0));
#endif

  // memory placeholders
  h_A1 = (int**)malloc(sizeof(int*) * devCount);
  h_B1 = (int**)malloc(sizeof(int*) * devCount);
  h_C1 = (int**)malloc(sizeof(int*) * devCount);
  h_A2 = (int**)malloc(sizeof(int*) * devCount);
  h_B2 = (int**)malloc(sizeof(int*) * devCount);
  h_C2 = (int**)malloc(sizeof(int*) * devCount);
  d_A1 = (int**)malloc(sizeof(int*) * devCount);
  d_B1 = (int**)malloc(sizeof(int*) * devCount);
  d_C1 = (int**)malloc(sizeof(int*) * devCount);
  d_A2 = (int**)malloc(sizeof(int*) * devCount);
  d_B2 = (int**)malloc(sizeof(int*) * devCount);
  d_C2 = (int**)malloc(sizeof(int*) * devCount);

  // test for all GPU devices
  RUNTIME_API_CALL(cudaGetDeviceCount(&devCount));
  for (deviceNum = 0; deviceNum < devCount; deviceNum++) {
    // provide device info
    DRIVER_API_CALL(cuDeviceGet(&device, deviceNum));
    DRIVER_API_CALL(cuDeviceGetName(deviceName, 32, device));
    printf("Device Name: %s\n", deviceName);

    // set device
    RUNTIME_API_CALL(cudaSetDevice(deviceNum));

    // setup memory
    size_t size = N_ELEMENTS * sizeof(int);
    h_A1[deviceNum] = (int*)malloc(size);
    h_B1[deviceNum] = (int*)malloc(size);
    h_C1[deviceNum] = (int*)malloc(size);
    h_A2[deviceNum] = (int*)malloc(size);
    h_B2[deviceNum] = (int*)malloc(size);
    h_C2[deviceNum] = (int*)malloc(size);
    RUNTIME_API_CALL(cudaMalloc((void**)&d_A1[deviceNum], size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_B1[deviceNum], size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_C1[deviceNum], size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_A2[deviceNum], size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_B2[deviceNum], size));
    RUNTIME_API_CALL(cudaMalloc((void**)&d_C2[deviceNum], size));

    // create streams
    cudaStream_t stream1, stream2;
    RUNTIME_API_CALL(cudaStreamCreate(&stream1));
    RUNTIME_API_CALL(cudaStreamCreate(&stream2));

    // execute using the streams
    /*
    doPass(h_A1[deviceNum], h_B1[deviceNum], h_C1[deviceNum],
           d_A1[deviceNum], d_B1[deviceNum], d_C1[deviceNum], size, stream1);
    doPass(h_A2[deviceNum], h_B2[deviceNum], h_C2[deviceNum],
           d_A2[deviceNum], d_B2[deviceNum], d_C2[deviceNum], size, stream2);
    */
    RUNTIME_API_CALL(cudaMemcpyAsync(d_A1[deviceNum], h_A1[deviceNum], size, cudaMemcpyHostToDevice, stream1));
    RUNTIME_API_CALL(cudaMemcpyAsync(d_A2[deviceNum], h_A2[deviceNum], size, cudaMemcpyHostToDevice, stream2));
    RUNTIME_API_CALL(cudaMemcpyAsync(d_B1[deviceNum], h_B1[deviceNum], size, cudaMemcpyHostToDevice, stream1));
    RUNTIME_API_CALL(cudaMemcpyAsync(d_B2[deviceNum], h_B2[deviceNum], size, cudaMemcpyHostToDevice, stream2));
    
    VecAdd<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, 0, stream1>>>(d_A1[deviceNum], d_B1[deviceNum], d_C1[deviceNum], N_ELEMENTS);
    VecAdd<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK, 0, stream2>>>(d_A2[deviceNum], d_B2[deviceNum], d_C2[deviceNum], N_ELEMENTS);

    RUNTIME_API_CALL(cudaMemcpyAsync(h_C1[deviceNum], d_C1[deviceNum], size, cudaMemcpyDeviceToHost, stream1));
    RUNTIME_API_CALL(cudaMemcpyAsync(h_C2[deviceNum], d_C2[deviceNum], size, cudaMemcpyDeviceToHost, stream2));

    // wait for completion
    RUNTIME_API_CALL(cudaDeviceSynchronize());

    // free memory
    free(h_A1[deviceNum]);
    free(h_B1[deviceNum]);
    free(h_C1[deviceNum]);
    free(h_A2[deviceNum]);
    free(h_B2[deviceNum]);
    free(h_C2[deviceNum]);
    cudaFree(d_A1[deviceNum]);
    cudaFree(d_B1[deviceNum]);
    cudaFree(d_C1[deviceNum]);
    cudaFree(d_A2[deviceNum]);
    cudaFree(d_B2[deviceNum]);
    cudaFree(d_C2[deviceNum]);

#if USE_CUPTI
    // Flush all remaining CUPTI buffers before resetting the device.
    // This can also be called in the cudaDeviceReset callback.
    cuptiActivityFlushAll(0);
#endif

    cudaDeviceReset();
  }
  return 0;
}
