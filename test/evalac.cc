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
    sst->lens = (int*)(g4c_ptr_add(sst->buf,
				   g4c_round_up(sst->count*sst->stride, G4C_PAGE_SIZE)));
    sst->ress = (int*)(g4c_ptr_add(sst->lens,
				   g4c_round_up(sst->count*sizeof(int), G4C_PAGE_SIZE)));

    sst->devstrs = (uint8_t*)sst->devbuf;
    sst->devlens = (int*)(g4c_ptr_add(sst->devbuf,
				      g4c_round_up(sst->count*sst->stride, G4C_PAGE_SIZE)));
    sst->devress = (int*)(g4c_ptr_add(sst->devlens,
				      g4c_round_up(sst->count*sizeof(int), G4C_PAGE_SIZE)));
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


void gpu_bench(g4c_acm_t *acm, eval_store *eitems, int ns)
{
    printf("GPU Bench, warm up: \n");

    timingval tv = timing_start();    
    g4c_h2d_async(eitems[ns-1].strs, eitems[ns-1].devstrs,
		  eitems[ns-1].count*eitems[ns-1].stride,
		  eitems[ns-1].stream);
    g4c_gpu_acm_match((g4c_acm_t*)acm->devmem, eitems[ns-1].count,
		      eitems[ns-1].devstrs, eitems[ns-1].stride, 0, 0,
		      eitems[ns-1].devress, 1, 0,
		      eitems[ns-1].stream, 0);
    g4c_d2h_async(eitems[ns-1].devress, eitems[ns-1].ress, eitems[ns-1].count*sizeof(int),
		  eitems[ns-1].stream);
    g4c_stream_sync(eitems[ns-1].stream);
    int64_t us = timing_stop(&tv);
    
    printf("Done warm up,      time %9ld us, BW %12.6lf MB/s, rate %12.6lf Mpkt/s\n",
	   us, ((double)eitems[ns-1].tlen)/(double)us,
	   ((double)eitems[ns-1].count)/(double)us);

    tv = timing_start();    
    for (int i=0; i<ns; i++) {
	g4c_h2d_async(eitems[i].strs, eitems[i].devstrs,
		      eitems[i].count*eitems[i].stride,
		      eitems[i].stream);
	g4c_gpu_acm_match((g4c_acm_t*)acm->devmem, eitems[i].count,
			  eitems[i].devstrs, eitems[i].stride, 0, 0,
			  eitems[i].devress, 1, 0,
			  eitems[i].stream, 0);
	g4c_d2h_async(eitems[i].devress, eitems[i].ress, eitems[i].count*sizeof(int),
		      eitems[i].stream);    
    }

    for (int i=0; i<ns; i++)
	g4c_stream_sync(eitems[i].stream);
    us = timing_stop(&tv);

    int ttlen = 0, tct=0;
    for (int i=0; i<ns; i++) {
	if (g_rand_lens)
	    ttlen += eitems[i].tlen;
	else
	    ttlen += eitems[i].count*eitems[i].stride;
	tct += eitems[i].count;
    }

    printf("Done benchmarking, time %9ld us, BW %12.6lf MB/s, rate %12.6lf Mpkt/s\n\n",
	   us/ns, ((double)ttlen)/(double)us,((double)tct)/(double)us);    
}

void cpu_bench(g4c_acm_t *acm, eval_store *eitems, int ns)
{
    printf("CPU Bench:\n");
    timingval tv = timing_start();
    for (int b=0; b<ns; b++) {
	for (int i=0; i<eitems[b].count; i++) {
	    eitems[b].ress[i] =
		g4c_cpu_acm_match(acm,
				  eitems[b].strs + i*eitems[b].stride,
				  eitems[b].lens[i]);
	}
    }
    int64_t us = timing_stop(&tv);

    for (int b=0; b<ns; b++)
	memset(eitems[b].ress, 0, sizeof(int)*eitems[b].count);

    int ttlen = 0, tct=0;
    for (int i=0; i<ns; i++) {
	if (g_rand_lens)
	    ttlen += eitems[i].tlen;
	else
	    ttlen += eitems[i].count*eitems[i].stride;
	
	tct += eitems[i].count;
    }

    printf("Done benchmarking, time %9ld us, BW %12.6lf MB/s, rate %12.6lf Mpkt/s\n\n",
	   us/ns, ((double)ttlen)/(double)us,((double)tct)/(double)us);    
}
  
int main(int argc, char *argv[])
{
    int mtype = 0;
    int nrstream = 4;
    int ptn_len = 16;
    int str_len = 1024;
    int nptns = 1024;
    int npkts = 1024;

    switch(argc) {
    case 8:
	g_rand_lens = atoi(argv[7])%2;
    case 7:
	mtype = atoi(argv[6])%2;
    case 6:
	nrstream = atoi(argv[5]);
    case 5:
	ptn_len = atoi(argv[4]);
    case 4:
	str_len = atoi(argv[3]);
    case 3:
	nptns = atoi(argv[2]);
    case 2:
	npkts = atoi(argv[1]);
	break;
    case 1:
	break;
    default:
	printf("Usage: %s [npkts] [nptns] [str_len] [ptn_len] [nrstream] [mtype] [rand_lens]\n",
	       argv[0]);
        return 0;
    }

    eval_init();

    printf("Eval AC: %d packets, %d patterns, %d max str len, %d max ptn len, %d streams\n",
	   npkts, nptns, str_len, ptn_len, nrstream);

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

    gpu_bench(acm, eval_items, nrstream);
    cpu_bench(acm, eval_items, nrstream);

    gpu_bench(acm, eval_items, nrstream);
    cpu_bench(acm, eval_items, nrstream);
    
    return 0;
}
