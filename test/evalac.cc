#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <iostream>
using namespace std;
#include "../g4c.h"
#include "utils.h"
#include "../ac.h"

static int g_rand_lens = 0;


static char **
gen_patterns(int np, int plen)
{
    size_t tsz = np*sizeof(char*) + np*(plen+1);

    char *p = (char*)malloc(tsz);
    if (!p)
	return 0;

    char *ptn = p+np*sizeof(char*);
    char **pp = (char**)p;
    srand((unsigned int)clock());

    for (int i=0; i<np; i++) {
	pp[i] = ptn;
	int j;
	for (j=0; j<plen; j++)
	    ptn[j] = (char)(rand()%60 + 'A');
	ptn[j] = (char)0;
	ptn += plen+1;
    }

    return pp;
}

typedef struct {
    void *buf;
    char *strs;
    int count;
    int stride;
    size_t bufsz;
    int *lens;
    int tlen;
} str_store;

static int
gen_strings(str_store *sst)
{
    sst->bufsz = (sst->stride + sizeof(int)) * sst->count;
    sst->buf = malloc(sst->bufsz);
    if (!sst->buf)
	return -ENOMEM;

    sst->strs = (char*)sst->buf;
    sst->lens = (int*)(g4c_ptr_add(sst->buf, sst->count*sst->stride));
    sst->tlen = 0;

    srand((unsigned int)clock());
    for (int i=0; i<sst->count; i++) {
	char *s = sst->strs + i*sst->stride;
	if (g_rand_lens)
	    sst->lens[i] = rand()%((sst->stride)-2) + 1;
	else
	    sst->lens[i] = sst->stride-1;
	sst->tlen += sst->lens[i];
	int j;
	for (j=0; j<sst->lens[i]; j++)
	    s[j] = (char)(rand()%60 + 'A');
	s[j] = (char)0;
    }

    return 0;   
}

static int g_nr_patterns = 8;
static int g_len_patterns = 16;
//static int g_nr_strings = 40960;
//static int g_str_stride = 1024;
static int g_acm_type=0;

void do_eval(int nstrs, int strstride)
{
    char** ptns = gen_patterns(g_nr_patterns, g_len_patterns);

    str_store sst;
    sst.count = nstrs;
    sst.stride = strstride;
    gen_strings(&sst);

    ac_machine_t acm;
    ac_build_machine(&acm, ptns, g_nr_patterns, 0);

    uint32_t *hress = (uint32_t*)g4c_alloc_page_lock_mem(
	sst.count*AC_ALPHABET_SIZE*sizeof(uint32_t));	

    {
	// CPU
	timingval tv = timing_start();
	for (int i=0; i<sst.count; i++) {
	    char *str = sst.strs + i*sst.stride;
	    ac_match(str, sst.lens[i], hress+i*AC_ALPHABET_SIZE, 0,
		     &acm);
	}
	int64_t ctv = timing_stop(&tv);
	printf("CPU size %dKB, time us %ld, rate %.3lfMB/s \n",
	       (sst.tlen)>>10, ctv,
	       ((double)(sst.tlen)/(double)ctv)
	    );
    }

    char *hstrs = (char*)g4c_alloc_page_lock_mem(sst.bufsz);
    memcpy(hstrs, sst.buf, sst.bufsz);
    char *dstrs = (char*)g4c_alloc_dev_mem(sst.bufsz);
    int *dlens =  (int*)g4c_ptr_add(dstrs, sst.count*sst.stride);
    uint32_t *dress = (uint32_t*)g4c_alloc_dev_mem(
	sst.count*AC_ALPHABET_SIZE*sizeof(uint32_t));

#define NRSTREAMS 2
    int streams[NRSTREAMS];
    for (int i=0; i<NRSTREAMS; i++)
	streams[i] = g4c_alloc_stream();
    ac_dev_machine_t *pdacm = 0;
    
    ac_prepare_gmatch(&acm, &pdacm, streams[0]);
    g4c_stream_sync(streams[0]);
    
    timingval tv = timing_start();
    for (int i=0; i<NRSTREAMS; i++) {
	g4c_h2d_async(hstrs, dstrs, sst.bufsz, streams[i]);
	ac_gmatch2(dstrs, sst.count, sst.stride, dlens,
		   dress, pdacm, streams[i], g_acm_type);
	g4c_d2h_async(dress, hress, sst.count*sizeof(int), streams[i]);
	// ac_gmatch_finish(sst.count, dress, hress, streams[i]);
    }
    for (int i=0; i<NRSTREAMS; i++)
	g4c_stream_sync(streams[i]);    
    int64_t usec = timing_stop(&tv)/NRSTREAMS;
    
    printf("GPU size %dKB, time us %ld, rate %.3lfMB/s \n",
	   (sst.tlen)>>10, usec,
	   ((double)(sst.tlen)/(double)usec)
	);

    
    free(ptns);
    free(sst.buf);
    g4c_free_host_mem(hress);
    g4c_free_host_mem(hstrs);
    g4c_free_dev_mem(dstrs);
    g4c_free_dev_mem(dress);
    for (int i=0; i<NRSTREAMS; i++)
	g4c_free_stream(streams[i]);
    ac_free_dev_acm(&pdacm);
    
    ac_release_machine(&acm);
}
  
int main(int argc, char *argv[])
{
    eval_init();

    if(argc > 1)
	g_acm_type = atoi(argv[1]);
    if(argc > 2)
	g_rand_lens = atoi(argv[2]);

    int nrs[] = { 1<<10, 1<<12, 1<<13, 1<<14};
    int strides[] = { 32, 64, 128, 256 };

    printf("warm up test:\n");
    do_eval(nrs[0], strides[0]);
    cout<<endl;

    for (int nr = 0; nr < sizeof(nrs)/sizeof(int); nr++) {
	for (int stride = 0;
	     stride < sizeof(strides)/sizeof(int); stride++) {
	    do_eval(nrs[nr], strides[stride]);
	    cout<<endl;
	}
    }

    for (int nr = 0; nr < sizeof(nrs)/sizeof(int); nr++) {
	for (int stride = 0;
	     stride < sizeof(strides)/sizeof(int); stride++) {
	    do_eval(nrs[nr]*10, strides[stride]);
	    cout<<endl;
	}
    }
    
    g4c_exit();
    return 0;
}
