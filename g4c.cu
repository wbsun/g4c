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
		return G4C_SENOTEXIST;
	}
}

#ifndef NR_STREAM
#define NR_STREAM 32
#endif

static cudaStream_t streams[NR_STREAM+1];
static int stream_uses[NR_STREAM+1];

#define csc(...) _cuda_safe_call(__VA_ARGS__, __FILE__, __LINE__)
static cudaError_t
_cuda_safe_call(cudaError_t e, const char *file, int line) {
    if (e!=cudaSuccess) {
	fprintf(stderr, "g4c Error: %s %d %s\n",
		file, line, cudaGetErrorString(e));
	cudaDeviceReset();
	abort();
    }
    return e;
}


extern "C" int
g4c_init(void) {
	int i;

	// Enable memory map, and spin CPU thread when waiting for sync to
	// decrease latency.
	csc( cudaSetDeviceFlags(
		     cudaDeviceScheduleSpin|cudaDeviceMapHost ) );

	// Create streams
	for (i=1; i<NR_STREAM+1; i++) {
		csc( cudaStreamCreate(&streams[i]) );
		stream_uses[i] = 0;
	}

	csc( cudaGetLastError() );

	return 0;
}

extern "C" void
g4c_exit(void) {
	int i;

	for (i=1; i<NR_STREAM+1; i++) {
		csc( cudaStreamDestroy(streams[i]) );
	}
}

extern "C" int
g4c_stream_done(int s) {
	cudaError_t e = cudaStreamQuery(streams[s]);
	if (e == cudaSuccess) {
		return 1;
	} else if (e != cudaErrorNotReady) {
		csc(e);
	}

	return 0;
}

extern "C" int
g4c_stream_sync(int s) {
	csc( cudaStreamSynchronize(streams[s]) );
	return 0;
}

extern "C" int
g4c_h2d_async(void *h, void *d, size_t sz, int s)
{
	csc( cudaMemcpyAsync(d, h, sz, cudaMemcpyHostToDevice, streams[s]) );
	return 0;
}

extern "C" int
g4c_d2h_async(void *d, void *h, size_t sz, int s)
{
	csc( cudaMemcpyAsync(h, d, sz, cudaMemcpyDeviceToHost, streams[s]) );
	return 0;
}

extern "C" int
g4c_dev_memset(void *d, int val, size_t sz, int s)
{
	csc( cudaMemsetAsync(d, val, sz, streams[s]) );
	return 0;
}

extern "C" int
g4c_alloc_stream()
{
	for (int i=1; i<NR_STREAM; i++) {
		if (stream_uses[i] == 0) {
			stream_uses[i] = 1;
			return i;
		}
	}

	return 0;
}

extern "C" void
g4c_free_stream(int s)
{
	stream_uses[s] = 0;
}

// Memory management functions.
extern "C" void *
g4c_alloc_page_lock_mem(size_t sz)
{
	return 0;
}

extern "C" void
g4c_free_page_lock_mem(void *p, size_t sz)
{
}

extern "C" void *
g4c_alloc_dev_mem(size_t sz)
{
	return 0;
}

extern "C" void
g4c_free_dev_mem(void *p, size_t sz)
{
}

// End of file.
