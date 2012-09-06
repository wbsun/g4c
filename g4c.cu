#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include "g4c.h"

static const char *G4C_SEMCPY = "GPU memory copy failed";
static const char *G4C_SEKERNEL = "GPU kernel launch or execution failed";
static const char *G4C_SENOTEXIST = "no such error";

extern "C" const char *g4c_strerror(int err) {
	switch(err) {
	case G4C_EMCPY:
		return G4C_SEMCPY;
	case G4C_EKERNEL:
		return G4C_SEKERNEL;
	default:
		return G4C_SENOEXIST;
	}
}

#ifndef NR_STREAM
#define NR_STREAM 8
#endif

static cudaStream_t streams[NR_STREAM];

#define csc(...) _cuda_safe_call(__VA_ARGS__, __FILE__, __LINE__)
static cudaError_t _cuda_safe_call(cudaError_t e, const char *file, int line) {
    if (e!=cudaSuccess) {
	fprintf(stderr, "g4c Error: %s %d %s\n",
		file, line, cudaGetErrorString(e));
	cudaThreadExit();
	abort();
    }
    return e;
}


__global__ void stuff_kernel(void *in, void *out, int n) {
	int *iin = (int*)in;
	int *iout = (int*)out;

	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	iout[tid] = iin[tid] + 10;
}


extern "C" int g4c_init(void) {
	int i;

	// Enable memory map, and spin CPU thread when waiting for sync to
	// decrease latency.
	csc( cudaSetDeviceFlags( cudaDeviceScheduleSpin|cudaDeviceMapHost ) );

	// Create streams
	for (i=0; i<NR_STREAM; i++) {
		csc( cudaStreamCreate(&streams[i]) );
	}

	return 0;
}

extern "C" void g4c_exit(void) {
	int i;

	for (i=0; i<NR_STREAM; i++) {
		csc( cudaStreamDestroy(streams[i]) );
	}
}

extern "C" void* g4c_malloc(size_t sz) {
	void *p;
	
	csc( cudaHostAlloc(&p, sz,
			   cudaHostAllocPortable|cudaHostAllocMapped) );

	return p;
}

extern "C" void g4c_free(void *p) {
	csc( cudaFreeHost(p) );
}

extern "C" int g4c_do_stuff_sync(void *in, void *out, int n) {
	stuff_kernel<<<n/32, 32>>>(in, out, n);
	csc( cudaThreadSynchronize() );

	return 0;
}

extern "C" int g4c_do_stuff_async(void *in, void *out, int n, g4c_async_t *adata) {
	adata->stream = 0;

	stuff_kernel<<<n/32, 32, 0, streams[0]>>>(in, out, n);

	return 0;
}

extern "C" int g4c_check_async_done(g4c_async_t *adata) {
	cudaError_t e = cudaStreamQuery(streams[adata->stream]);
	if (e == cudaSuccess) {
		return 1;
	} else if (e != cudaErrorNotReady) {
		csc(e);
	}

	return 0;
}
