#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>
#include <iostream>
using namespace std;

#include "utils.h"
#include "../g4c.h"
#include "../ac.h"

static int
eval_init() {
    int r = g4c_init(G4C_DEFAULT_NR_STREAMS,
		     G4C_DEFAULT_MEM_SIZE,
		     G4C_DEFAULT_WCMEM_SIZE,
		     G4C_DEFAULT_MEM_SIZE+G4C_DEFAULT_WCMEM_SIZE);
    return r;
}


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

    srand((unsigned int)clock());
    for (int i=0; i<sst->count; i++) {
	char *s = sst->strs + i*sst->stride;
	// sst->lens[i] = rand()%((sst->stride)-2) + 1;
	int j;
	sst->lens[i] = sst->stride-1;
	for (j=0; j<sst->lens[i]; j++)
	    s[j] = (char)(rand()%60 + 'A');
	s[j] = (char)0;
    }

    return 0;   
}

static int g_nr_patterns = 8;
static int g_len_patterns = 4;
//static int g_nr_strings = 40960;
//static int g_str_stride = 1024;

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
	       (nstrs*strstride)>>10, ctv,
	       ((double)(nstrs*strstride)/(double)ctv)
	    );
    }

    char *hstrs = (char*)g4c_alloc_wc_mem(sst.bufsz);
    memcpy(hstrs, sst.buf, sst.bufsz);
    char *dstrs = (char*)g4c_alloc_dev_mem(sst.bufsz);
    int *dlens =  (int*)g4c_ptr_add(dstrs, sst.count*sst.stride);
    uint32_t *dress = (uint32_t*)g4c_alloc_dev_mem(
	sst.count*AC_ALPHABET_SIZE*sizeof(uint32_t));

    int stream = g4c_alloc_stream();
    ac_dev_machine_t *pdacm = 0;
    
    ac_prepare_gmatch(&acm, &pdacm, stream);
    g4c_stream_sync(stream);
    
    timingval tv = timing_start();
    g4c_h2d_async(hstrs, dstrs, sst.bufsz, stream);
    ac_gmatch(dstrs, sst.count, sst.stride, dlens,
	      dress, pdacm, stream);
   ac_gmatch_finish(sst.count, dress, hress, stream);
    g4c_stream_sync(stream);    
    int64_t usec = timing_stop(&tv);
    
    printf("GPU size %dKB, time us %ld, rate %.3lfMB/s \n",
	   (nstrs*strstride)>>10, usec,
	   ((double)(nstrs*strstride)/(double)usec)
	);

    
    free(ptns);
    free(sst.buf);
    g4c_free_host_mem(hress);
    g4c_free_host_mem(hstrs);
    g4c_free_dev_mem(dstrs);
    g4c_free_dev_mem(dress);
    g4c_free_stream(stream);
    ac_free_dev_acm(&pdacm);
    
    ac_release_machine(&acm);
}
  
int main(int argc, char *argv[])
{
    eval_init();

    int nrs[] = { 1<<7, 1<<8, 1<<9, 1<<10, 1<<11, 1<<12};
    int strides[] = { 32, 64, 128, 256 };

    for (int nr = 0; nr < sizeof(nrs)/sizeof(int); nr++) {
	for (int stride = 0; stride < sizeof(strides)/sizeof(int); stride++) {
	    do_eval(nrs[nr], strides[stride]);
	    cout<<endl;
	}
    }

    for (int nr = 0; nr < sizeof(nrs)/sizeof(int); nr++) {
	for (int stride = 0; stride < sizeof(strides)/sizeof(int); stride++) {
	    do_eval(nrs[nr]*10, strides[stride]);
	    cout<<endl;
	}
    }
    
    g4c_exit();
    return 0;
}
