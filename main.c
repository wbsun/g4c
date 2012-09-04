#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include "g4c.h"

int main(int argc, char *argv[]) {
	int i, n = 4096*1024;
	int *in, *out;
	g4c_async_t adata;

	printf("initializing device...\n");
	g4c_init();

	printf("allocating memory...\n");
	in = g4c_malloc(sizeof(int)*n);
	out = g4c_malloc(sizeof(int)*n);
	if (!in || !out) {
		fprintf(stderr, "CUDA memory allocation faile!");
		abort();
	}

	printf("initialize buffers...\n");
	for (i=0; i<n; i++) {
		in[i] = i+1000;
		out[i] = 0;
	}

	printf("starting doing stuff...\n");
	// g4c_do_stuff_sync(in, out, n);

	g4c_do_stuff_async(in, out, n, &adata);

	while(!g4c_check_async_done(&adata)) {
		printf("stuff not done, sleep for a while...\n");
		if (usleep(3) == -1) {
			perror("usleep error");
		}
	}

	printf("stuff done, dump results...\n");
	for (i=0; i<n; i+=4096) {
		printf("%-4d ", out[i]);
		if ((i+1)%(32*4096) == 0)
			printf("\n");
	}

	printf("free buffers...\n");
	g4c_free(in);
	g4c_free(out);

	printf("all done, exit!\n");
	return 0;
}
	
