#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <errno.h>
#include <iostream>
using namespace std;
#include "../g4c.h"
#include "utils.h"
#include "../g4c_ac.h"

static int g_rand_lens = 0;


static char **
gen_patterns(int np, int plen)
{
    struct timeval tv;
    gettimeofday(&tv, 0);

    srandom((unsigned)(tv.tv_usec));

    size_t tsz = np*sizeof(char*) + np*(plen);

    char *p = (char*)malloc(tsz);
    if (!p)
	return 0;

    char *ptn = p+np*sizeof(char*);
    char **pp = (char**)p;

    for (int i=0; i<np; i++) {
	pp[i] = ptn;
	int j;
	int mylen = (random()%(plen-4)) + 3;
	for (j=0; j<mylen; j++)
	    ptn[j] = (char)(random()%60 + 'A');
	ptn[j] = (char)0;
	ptn += plen;
    }

    return pp;    
}

typedef struct {
    void *buf;
    uint8_t *strs;
    int count;
    int stride;
    size_t bufsz;
    int *lens;
    int tlen;
    int *ress;

    void *devbuf;
    uint8_t *devstrs;
    int* devlens;
    int *devress;

    int stream;
} eval_store;

static int
gen_eval_store(eval_store *sst, int str_stride, int nstrs)
{
    sst->count = nstrs;
    sst->stride = str_stride;
    sst->bufsz = g4c_round_up(sst->stride*sst->count, G4C_PAGE_SIZE) +
	g4c_round_up(sizeof(int) * sst->count, G4C_PAGE_SIZE)*2;
    sst->buf = g4c_alloc_page_lock_mem(sst->bufsz);
    if (!sst->buf)
	return -ENOMEM;
    sst->devbuf = g4c_alloc_dev_mem(sst->bufsz);
    if (!sst->buf) {
	g4c_free_host_mem(sst->buf);
	return -ENOMEM;
    }

    sst->strs = (uint8_t*)sst->buf;
    sst->lens = (int*)(g4c_ptr_add(
			   sst->buf,
			   g4c_round_up(
			       sst->count*sst->stride, G4C_PAGE_SIZE)));
    sst->ress = (int*)(g4c_ptr_add(
			   sst->lens,
			   g4c_round_up(
			       sst->count*sizeof(int), G4C_PAGE_SIZE)));

    sst->devstrs = (uint8_t*)sst->devbuf;
    sst->devlens = (int*)(g4c_ptr_add(
			      sst->devbuf,
			      g4c_round_up(
				  sst->count*sst->stride, G4C_PAGE_SIZE)));
    sst->devress = (int*)(g4c_ptr_add(
			      sst->devlens,
			      g4c_round_up(
				  sst->count*sizeof(int), G4C_PAGE_SIZE)));
    sst->tlen = 0;

    for (int i=0; i<sst->count; i++) {
	uint8_t *s = sst->strs + i*sst->stride;
	if (g_rand_lens)
	    sst->lens[i] = random()%((sst->stride)-3) + 2;
	else
	    sst->lens[i] = sst->stride-1;
	sst->tlen += sst->lens[i];
	int j;
	for (j=0; j<sst->lens[i]; j++)
	    s[j] = (char)(random()%200)+1;
	s[j] = (char)0;
	sst->ress[i] = 0;
    }

    return 0;   
}

  
int main(int argc, char *argv[])
{
    int mtype = 0;
    int nrstream = 4;
    int ptn_len = 16;
    int str_len = 1024;
    int nptns = 1024;
    int npkts = 1024;

    int nrpkts[] = {16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int nszs = sizeof(nrpkts)/sizeof(int);
    npkts = nrpkts[nszs-1];

    switch(argc) {
    case 7:
	g_rand_lens = atoi(argv[6])%2;
    case 6:
	mtype = atoi(argv[5])%2;
    case 5:
	nrstream = atoi(argv[4]);
    case 4:
	ptn_len = atoi(argv[3]);
    case 3:
	str_len = atoi(argv[2]);
    case 2:
	nptns = atoi(argv[1]);
    case 1:
	break;
    default:
	printf("Usage: %s [nptns] [str_len] [ptn_len] "
	       "[nrstream] [mtype] [rand_lens]\n",
	       argv[0]);
        return 0;
    }

    eval_init();

    printf("Eval AC: %d patterns, %d max str len, %d max "
	   "ptn len, %d streams\n",
	   nptns, str_len, ptn_len, nrstream);

    printf("Generating patterns... ");
    char **ptns = gen_patterns(nptns, ptn_len);
    if (!ptns) {
	printf("Failed\n");
	return 0;
    } else
	printf("Done\n");

    printf("Generating ACM... ");
    int s = g4c_alloc_stream();
    g4c_acm_t *acm = g4c_create_matcher(ptns, nptns, 1, s);
    if (!acm) {
	printf("Failed\n");
	return 0;
    } else
	printf("Done\n");
    
    eval_store *eval_items = new eval_store[nrstream];
    if (!eval_items) {
	fprintf(stderr, "Out of mem for evaluation item array\n");
	return 0;
    }

    printf("Generating evaluation items... ");
    for (int i=0; i<nrstream; i++) {
	if (gen_eval_store(eval_items+i, str_len, npkts)) {
	    printf("failed on %d\n", i);
	    return 0;
	}	    
    }
    printf("Done\n");

    printf("Allocating streams... ");
    eval_items[0].stream = s;
    for (int i=1; i<nrstream; i++)
	eval_items[i].stream = g4c_alloc_stream();
    printf("Done\n");

    int ns = nrstream; // silly hack

//    printf("GPU Bench, warm up: \n");
    timingval tv = timing_start();    
    g4c_h2d_async(eval_items[ns-1].strs, eval_items[ns-1].devstrs,
		  eval_items[ns-1].count*eval_items[ns-1].stride,
		  eval_items[ns-1].stream);
    g4c_gpu_acm_match((g4c_acm_t*)acm->devmem,
		      eval_items[ns-1].count,
		      eval_items[ns-1].devstrs,
		      eval_items[ns-1].stride, 0, 0,
		      eval_items[ns-1].devress, 1, 0,
		      eval_items[ns-1].stream, mtype);
    g4c_d2h_async(eval_items[ns-1].devress,
		  eval_items[ns-1].ress,
		  eval_items[ns-1].count*sizeof(int),
		  eval_items[ns-1].stream);
    g4c_stream_sync(eval_items[ns-1].stream);
    int64_t us = timing_stop(&tv);    
    // printf("Done warm up,      time %9ld us, BW %12.6lf MB/s, "
	   // "rate %12.6lf Mpkt/s\n",
	   // us, ((double)eval_items[ns-1].tlen)/(double)us,
	   // ((double)eval_items[ns-1].count)/(double)us);

    int64_t *gtimes = new int64_t[nszs];
    int64_t *ctimes = new int64_t[nszs];
    int *blens = new int[nszs];

    for (int b=0; b<nszs; b++) {
	tv = timing_start();    
	for (int i=0; i<ns; i++) {
	    g4c_h2d_async(eval_items[i].strs,
			  eval_items[i].devstrs,
			  nrpkts[b]*eval_items[i].stride,
			  eval_items[i].stream);
	    g4c_gpu_acm_match((g4c_acm_t*)acm->devmem,
			      nrpkts[b],
			      eval_items[i].devstrs,
			      eval_items[i].stride, 0, 0,
			      eval_items[i].devress, 1, 0,
			      eval_items[i].stream, 0);
	    g4c_d2h_async(eval_items[i].devress,
			  eval_items[i].ress,
			  nrpkts[b]*sizeof(int),
			  eval_items[i].stream);    
	}

	for (int i=0; i<ns; i++)
	    g4c_stream_sync(eval_items[i].stream);
	gtimes[b] = timing_stop(&tv)/ns;

	int ttlen = 0;
	for (int i=0; i<ns; i++) {
	    if (g_rand_lens) {
		for (int k=0; k<nrpkts[b]; k++)
		    ttlen += eval_items[i].lens[k];
	    }
	    else
		ttlen += nrpkts[b]*(eval_items[i].stride);
	}
	blens[b] = ttlen/ns;
    }

    for (int b=0; b<nszs; b++) {
	tv = timing_start();
	for (int st=0; st<ns; st++) {
	    for (int i=0; i<nrpkts[b]; i++) {
		eval_items[st].ress[i] =
		    g4c_cpu_acm_match(
			acm,
			eval_items[st].strs + i*eval_items[st].stride,
			eval_items[st].lens[i]+1);
	    }
	}
	ctimes[b] = timing_stop(&tv);
    }

    for (int b=0; b<nszs; b++) {
	printf("Done GPU, pkts %6d, BW %12.5lf MB/s, rate %12.6lf Mpps\n",
	       nrpkts[b], ((double)blens[b])/(double)gtimes[b],
	       ((double)nrpkts[b])/(double)gtimes[b]);
    }

    for (int b=0; b<nszs; b++) {
	printf("Done CPU, pkts %6d, BW %12.5lf MB/s, rate %12.6lf Mpps\n",
	       nrpkts[b], ((double)(blens[b]*ns))/(double)ctimes[b],
	       ((double)(nrpkts[b]*ns))/(double)ctimes[b]);
    }
 
    
    return 0;
}
