#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "g4c.h"
#include "g4c_cl.h"
#include "utils.h"

#define PKT_LEN 16

#include <sys/time.h>
static void
gen_rand_ptns(g4c_pattern_t *ptns, int n)
{
    struct timeval tv;
    gettimeofday(&tv, 0);

    srandom((unsigned)(tv.tv_usec));

    for (int i=0; i<n; i++) {
	int nbits = random()%5;
	ptns[i].nr_src_netbits = nbits*8;
	for (int j=0; j<nbits; j++)
	    ptns[i].src_addr = (ptns[i].src_addr<<8)|(random()&0xff);
	nbits = random()%5;
	ptns[i].nr_dst_netbits = nbits*8;
	for (int j=0; j<nbits; j++)
	    ptns[i].dst_addr = (ptns[i].dst_addr<<8)|(random()&0xff);
	ptns[i].src_port = random()%(PORT_STATE_SIZE<<1);
	if (ptns[i].src_port >= PORT_STATE_SIZE)
	    ptns[i].src_port -= PORT_STATE_SIZE*2;
	ptns[i].dst_port = random()%(PORT_STATE_SIZE<<1);
	if (ptns[i].dst_port >= PORT_STATE_SIZE)
	    ptns[i].dst_port -= PORT_STATE_SIZE*2;
	ptns[i].proto = random()%(PROTO_STATE_SIZE);
	ptns[i].idx = i;
    }
}

static void
gen_rand_pkts(uint8_t *pkts, int n)
{
    struct timeval tv;
    gettimeofday(&tv, 0);

    srandom((unsigned)(tv.tv_usec));

    for (int i=0; i<n; i++) {
	uint8_t *pkt = pkts + i*PKT_LEN;
	pkt[1] = random() & 0xff;
	for (int j=0; j<4; j++) {
	    pkt[4+j] = random() & 0xff;
	    pkt[8+j] = random() & 0xff;
	    pkt[12+j] = random() & 0xff;
	}
    }    
}

static void
gpu_bench(g4c_classifier_t *hgcl, g4c_classifier_t *dgcl,
	  uint8_t *hppkts[], uint8_t *dppkts[], int npkts,
	  int *hpress[], int *dpress[], uint32_t res_stride, uint32_t res_ofs,
	  int *streams, int ns)
{
    printf("GPU Bench, warm up: \n");

    timingval tv = timing_start();    
    g4c_h2d_async(hppkts[ns-1], dppkts[ns-1], npkts*PKT_LEN, stream[ns-1]);
    g4c_gpu_classify_pkts(dgcl, npkts, dppkts[ns-1], PKT_LEN, 1, 12, dpress[ns-1],
			  res_stride, res_ofs, stream[ns-1]);
    g4c_d2h_async(dpress[ns-1], hpress[ns-1], npkts*res_stride, stream[ns-1]);
    g4c_stream_sync(stream[ns-1]);
    int64_t us = timing_stop(&tv);
    
    printf("Done warm up, time %12ld us, rate %8.6lf Mops/s\n\n",
	   us, ((double)npkts)/(double)us);

    tv = timing_start();    
    for (int i=0; i<ns; i++) {
	g4c_h2d_async(hppkts[i], dppkts[i], npkts*PKT_LEN, stream[i]);
	g4c_gpu_classify_pkts(dgcl, npkts, dppkts[i], PKT_LEN, 1, 12, dpress[i],
			      res_stride, res_ofs, stream[i]);
	g4c_d2h_async(dpress[i], hpress[i], npkts*res_stride, stream[i]);
    }
    us = timing_stop(&tv);

    printf("Done benchmarking, time %12ld us, rate %8.6lf Mops/s\n\n",
	   us/ns, ((double)npkts*ns)/(double)us);    
}

static void
cpu_bench(g4c_classifier_t *gcl, uint8_t *ppkts[], int npkts,
	  int *press[], uint32_t res_stride, uint32_t res_ofs, int nbatch)
{
    printf("CPU Bench\n");
    timingval tv = timing_start();
    for (int b=0; b<nbatch; b++) {
	for (int i=0; i<npkts; i++) {
	    *(press[b] + i*res_stride + res_ofs) =
		g4c_cpu_classify_pkt(gcl, ppkts[b]+i*PKT_LEN+1);
	}
    }
    int64_t us = timing_stop(&tv);
    
    printf("Done benchmarking, time %12ld us, rate %8.6lf Mops/s\n\n",
	   us/nbatch, ((double)npkts*nbatch)/(double)us);    
}
	 
int main(int argc, char *argv[])
{
    int res_stride = 2;
    int res_ofs = 0;
    
    int nptns, npkts, nstream;
    nptns = 1024;
    npkts = (1<<20);
    nstream = 4;

    switch(argc) {    
    case 4:
	nstream = atoi(argv[3]);
    case 3:
	char snpkts[32];
	int unit = 1;
	strcpy(snpkts, argv[2]);
	switch(snpkts[strlen(snpkts)-1]) {
	case 'M':
	case 'm':
	    unit = (1<<20);
	    snpkts[strlen(snpkts)-1] = '\0';
	    break;
	case 'K':
	case 'k':
	    unit = (1<<10);
	    snpkts[strlen(snpkts)-1] = '\0';
	    break;
	default:
	    break;
	}

	npkts = atoi(snpkts)*unit;
    case 2:
	nptns = atoi(argv[1]);
	break;
    default:
	break;
    }

    printf("CL bench: %d patterns, %d packets, %d streams\n",
	   nptns, npkts, nstream);

    g4c_classifier_t *gcl;
    g4c_pattern_t *ptns = new g4c_pattern_t[nptns];
    uint8_t **hppkts = new uint8_t*[nstream];
    uint8_t **dppkts = new uint8_t*[nstream];
    int **hpress = new int*[nstream];
    int **dpress = new int*[nstream];
    int *streams = new int[nstream];    

    assert(ptns && hppkts && dppkts && hpress && dpress && streams);

    printf("Generating patterns... ");
    gen_rand_ptns(ptns, nptns);
    printf(" Done.\n");
    
    eval_init();

    for (int i=0; i<nstream; i++) {
	hppkts[i] = g4c_alloc_page_lock_mem(npkts*PKT_LEN);
	dppkts[i] = g4c_alloc_dev_mem(npkts*PKT_LEN);
	hpress[i] = g4c_alloc_page_lock_mem(npkts*sizeof(int)*res_stride);
	dpress[i] = g4c_alloc_dev_mem(npkts*sizeof(int)*res_stride);
	stream[i] = g4c_alloc_stream();
	assert(hppkts[i] && dppkts[i] && hpress[i] && dpress[i] && stream[i]);

	printf("Generating %d-th packets batch... ", i);
	gen_rand_pkts(hppkts[i], npkts);
	printf(" Done.\n");
    }

    printf("Build classifier... ");
    gcl = g4c_create_classifier(ptns, nptns, 1, stream[0]);
    if (gcl)
	printf(" Done.\n");
    else {
	printf(" Failed, abort\n");
	return 0;
    }

    gpu_bench(gcl, (g4c_classifier_t*)gcl->devmem,
	      hppkts, dppkts, npkts, hpress, dpress, res_stride, res_ofs, streams, nstream);
    cpu_bench(gcl, hppkts, npkts, hpress, res_stride, res_ofs, nstream);

    gpu_bench(gcl, (g4c_classifier_t*)gcl->devmem,
	      hppkts, dppkts, npkts, hpress, dpress, res_stride, res_ofs, streams, nstream);
    cpu_bench(gcl, hppkts, npkts, hpress, res_stride, res_ofs, nstream);

    return 0;
}
