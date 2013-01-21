#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include "../g4c.h"
#include "../g4c_cl.h"
#include "utils.h"

#define PKT_LEN 16

int g_debug = 0;

#include <sys/time.h>
static void
gen_rand_ptns(g4c_pattern_t *ptns, int n)
{
    struct timeval tv;
    gettimeofday(&tv, 0);

    srandom((unsigned)(tv.tv_usec));

    for (int i=0; i<n; i++) {
	int nbits = random()%5;
	if (random()%3) {
	    ptns[i].nr_src_netbits = nbits*8;
	    for (int j=0; j<nbits; j++)
		ptns[i].src_addr = (ptns[i].src_addr<<8)|(random()&0xff);
	} else
	    ptns[i].nr_src_netbits = 0;
	
	if (random()%3) {
	    nbits = random()%5;
	    ptns[i].nr_dst_netbits = nbits*8;
	    for (int j=0; j<nbits; j++)
		ptns[i].dst_addr = (ptns[i].dst_addr<<8)|(random()&0xff);
	} else
	    ptns[i].nr_dst_netbits = 0;
	
	if (random()%3) {
	    ptns[i].src_port = random()%(PORT_STATE_SIZE<<1);
	    if (ptns[i].src_port >= PORT_STATE_SIZE)
		ptns[i].src_port -= PORT_STATE_SIZE*2;
	} else
	    ptns[i].src_port = -1;
	
	if (random()%3) {
	    ptns[i].dst_port = random()%(PORT_STATE_SIZE<<1);
	    if (ptns[i].dst_port >= PORT_STATE_SIZE)
		ptns[i].dst_port -= PORT_STATE_SIZE*2;
	} else
	    ptns[i].dst_port = -1;
	
	if (random()%3) {
	    ptns[i].proto = random()%(PROTO_STATE_SIZE);
	} else
	    ptns[i].proto = -1;
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

	 
int main(int argc, char *argv[])
{
    int res_stride = 1;
    int res_ofs = 0;
    
    int nptns, npkts, nstream;
    nptns = 1024;
    nstream = 4;

    char snpkts[32];
    int unit = 1;

    int nrpkts[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int nszs = sizeof(nrpkts)/sizeof(int);
    npkts = nrpkts[nszs-1];

    switch(argc) {    
    case 3:
	nstream = atoi(argv[2]);
    case 2:
	nptns = atoi(argv[1]);
	break;
    default:
	break;
    }

    printf("CL bench: %d patterns, %d streams\n",
	   nptns, nstream);

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

    printf("Initialize G4C... ");
    eval_init();
    char clv = 'v';
    int clvv = 0;
    g4c_cl_init(1, &clv, &clvv);
    printf(" Done.\n");

    printf("Preparing data... ");
    for (int i=0; i<nstream; i++) {
	hppkts[i] = (uint8_t*)g4c_alloc_page_lock_mem(npkts*PKT_LEN);
	dppkts[i] = (uint8_t*)g4c_alloc_dev_mem(npkts*PKT_LEN);
	hpress[i] = (int*)g4c_alloc_page_lock_mem(npkts*sizeof(int)*res_stride);
	dpress[i] = (int*)g4c_alloc_dev_mem(npkts*sizeof(int)*res_stride);
	streams[i] = g4c_alloc_stream();
	assert(hppkts[i] && dppkts[i] && hpress[i] && dpress[i] && streams[i]);

	gen_rand_pkts(hppkts[i], npkts);
    }
    printf(" Done.\n");

    printf("Build classifier... ");
    gcl = g4c_create_classifier(ptns, nptns, 1, streams[0]);
    if (gcl)
	printf(" Done.\n");
    else {
	printf(" Failed, abort\n");
	return 0;
    }


    // printf("GPU Bench, warm up: \n");

    timingval tv = timing_start();    
    g4c_h2d_async(hppkts[nstream-1],
		  dppkts[nstream-1],
		  npkts*PKT_LEN,
		  streams[nstream-1]);
    g4c_gpu_classify_pkts((g4c_classifier_t*)gcl->devmem,
			  npkts,
			  dppkts[nstream-1],
			  PKT_LEN, 0, 12,
			  dpress[nstream-1],
			  res_stride, res_ofs,
			  streams[nstream-1]);
    g4c_d2h_async(dpress[nstream-1],
		  hpress[nstream-1],
		  npkts*res_stride,
		  streams[nstream-1]);
    g4c_stream_sync(streams[nstream-1]);
    int64_t us = timing_stop(&tv);
    
    // printf("Done warm up, pkts %6d     time %9ld us, rate %12.6lf Mops/s\n",
	   // npkts, us, ((double)npkts)/(double)us);

    int64_t *gtimes = new int64_t[nszs];
    int64_t *ctimes = new int64_t[nszs];

    for (int b=0; b<nszs; b++) {
	tv = timing_start();    
	for (int i=0; i<nstream; i++) {
	    g4c_h2d_async(hppkts[i],
			  dppkts[i],
			  nrpkts[b]*PKT_LEN,
			  streams[i]);
	    g4c_gpu_classify_pkts((g4c_classifier_t*)gcl->devmem,
				  nrpkts[b],
				  dppkts[i],
				  PKT_LEN, 0, 12,
				  dpress[i],
				  res_stride, res_ofs,
				  streams[i]);
	    g4c_d2h_async(dpress[i],
			  hpress[i],
			  nrpkts[b]*res_stride,
			  streams[i]);
	}

	for (int i=0; i<nstream; i++)
	    g4c_stream_sync(streams[i]);

	gtimes[b] = timing_stop(&tv)/nstream;
    }


    for (int i=0; i<0; i++) {
	for (int j=0; j<nrpkts[nszs-1]; j++) {
	    *(hpress[i] + j*res_stride + res_ofs) =
		g4c_cpu_classify_pkt(gcl, hppkts[i]+j*PKT_LEN+1);
	}
    }
    
    for (int b=0; b<nszs; b++) {
	tv = timing_start();
	for (int i=0; i<nstream; i++) {
	    for (int j=0; j<nrpkts[b]; j++) {
		*(hpress[i] + j*res_stride + res_ofs) =
		    g4c_cpu_classify_pkt(gcl, hppkts[i]+j*PKT_LEN+1);
	    }
	}
	ctimes[b] = timing_stop(&tv);
    }

    for (int b=0; b<nszs; b++) {
	printf("Done GPU, pkts %6d, time %9ld us, rate %12.6lf Mops/s\n",
	       nrpkts[b], gtimes[b], ((double)nrpkts[b])/(double)gtimes[b]);
    }
    for (int b=0; b<nszs; b++) {
	printf("Done CPU, pkts %6d, time %9ld us, rate %12.6lf Mops/s\n",
	       nrpkts[b], ctimes[b]/nstream,
	       ((double)nrpkts[b]*nstream)/(double)ctimes[b]);
    }    

    return 0;
}
