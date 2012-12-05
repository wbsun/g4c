#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <errno.h>

#include "utils.h"
#include <g4c.h>
#include <ac.h>

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
	for (int j=0; j<plen; j++)
	    ptn[j] = (char)(rand()/60 + 'A');
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
    sst->lens = (int*)(g4c_ptr_add(sst->buf, sst->count*stride));

    srand((unsigned int)clock());
    for (int i=0; i<sst->count; i++) {
	char *s = sst->strs + i*sst->stride;
	sst->len[i] = rand()/((stride)-2) + 1;
	for (int j=0; j<sst->len[i]; j++)
	    s[j] = (char)(rand()/60 + 'A');
	s[j] = (char)0;
    }

    return 0;   
}

static int g_nr_patterns = 32;
static int g_len_patterns = 4;
static int g_nr_strings = 4096;
static int g_str_stride = 32;
  
int main(int argc, char *argv[])
{
    eval_init();

    char** ptns = gen_patterns(g_nr_patterns, g_len_patterns);

    str_store sst;
    sst.count = g_nr_strings;
    sst.stride = g_str_stride;
    gen_strings(&sst);

    ac_machine_t acm;
    ac_build_machine(&acm, ptns, g_nr_patterns, 0);

    char *dstrs = (char*)g4c_alloc_wc_mem(sst.bufsz);
    int *dlens =  (int*)g4c_ptr_add(dstrs, sst.count*sst.stride);
    uint32_t *dress = (uint32_t*)g4c_alloc_dev_mem(
	sst.count*AC_ALPHABET_SIZE*sizeof(uint32_t));

    uint32_t *hress = (uint32_t*)g4c_alloc_page_lock_mem(
	sst.count*AC_ALPHABET_SIZE*sizeof(uint32_t));

    int stream = g4c_alloc_stream();
    ac_dev_machine_t *pdacm = 0;

    g4c_h2d_async(sst.buf, dstrs, sst.count*sst.stride, stream);
    ac_prepare_gmatch(&acm, &pdacm, stream);
    g4c_stream_sync(stream);
    
    timingval tv = timing_start();
    ac_gmatch(dstrs, sst.count, sst.stride, dlens,
	      dress, pdacm, stream);
    ac_gmatch_finish(sst.count, dress, hress, stream);
    g4c_stream_sync(stream);    
    int64_t usec = timing_stop(&tv);
    
    printf("size %d, time us %l\n", g_nr_strings*g_str_stride, usec);
    ac_release_machine(acm);
    g4c_exit();
    return 0;
}
