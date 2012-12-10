#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include "../g4c.h"
#include "../lookup.h"

struct rtlu_eval {
    uint32_t *haddrs;
    int n;
    uint32_t *daddrs;
    g4c_lpmtree_t *hlpmt;
    g4c_lpmtree_t *dlpmt;
    uint8_t *hports;
    uint8_t *dports;
};

static int
gen_ip_addrs(uint32_t **addrs, int n)
{
    if (!*addrs) {
	*addrs = (uint32_t*)g4c_alloc_page_lock_mem(sizeof(uint32_t)*n);
	if (!*addrs)
	    return -ENOMEM;
    }
    srand((unsigned int)clock());
    for (int i=0; i<n; i++) {
	for (int j=0; j<4; j++)
	    ((unsigned char*)((*addrs)+i))[j] =
		(unsigned char)(rand()%256);
    }
    return 0;
}

static int
gen_rt_entries(g4c_rte_t **ents, int n)
{
    if (!*ents) {
	*ents = (g4c_rte_t *)malloc(sizeof(g4c_rte_t)*n);
	if (!*ents)
	    return -ENOMEM;
    }
    srand((unsigned int)clock());
    for (int i=0; i<n; i++) {
	int x = rand()%3;
	(*ents)[i].nnetbits = 8*(x+1);
	(*ents)[i].mask = 0xffffffff<<(32-(*ents)[i].nnetbits);
	switch(x) {
	case 2:
	    ((unsigned char*)&((*ents)[i].addr))[2] =
		(unsigned char)(rand()%256);
	case 1:
	    ((unsigned char*)&((*ents)[i].addr))[1] =
		(unsigned char)(rand()%256);
	case 0:
	    ((unsigned char*)&((*ents)[i].addr))[0] =
		(unsigned char)(rand()%256);
	}
	(*ents)[i].addr &= (*ents)[i].mask;
	(*ents)[i].port = (uint8_t)rand()%128;
    }

    return 0;	
}

static int
prepare_eval_item(rtlu_eval *item, int n, int nrt, int s)
{
    g4c_rte_t *ents;
    gen_rt_entries(&ents, nrt);
    
    g4c_lpmtree_t *lpmt = g4c_build_lpm_tree(ents, nrt, 2, 0);

    size_t tsz = sizeof(g4c_lpm_tree)+sizeof(g4c_lpmnode2b_t)*lpmt->nnodes;
    item->hlpmt = (g4c_lpmtree_t*)g4c_alloc_page_lock_mem(tsz);
    item->dlpmt = (g4c_lpmtree_t*)g4c_alloc_dev_mem(tsz);
    memcpy(item->hlpmt, lpmt, tsz);
    g4c_h2d_async(item->hlpmt, item->dlpmt, s);
    free(lpmt);

    item->n = n;
    gen_ip_addrs(&item->haddrs, n);
    item->daddrs = (uint32_t*)g4c_alloc_dev_mem(sizeof(uint32_t)*n);
    item->hports = (uint8_t*)g4c_alloc_page_lock_mem(sizeof(uint8_t)*n);
    item->dports = (uint8_t*)g4c_alloc_dev_mem(sizeof(uint8_t)*n);

    return 0;
}


int g_nr_stream = 4;

int main(int argc, char *argv[])
{
    eval_init();
    
    rtlu_eval *items = (rtlu_eval*)malloc(sizeof(rtlu_eval)*g_nr_stream);
    int *streams = (int*)malloc(sizeof(int)*g_nr_stream);

    int nrpkts[] = { 1<<10, 1<<12, 1<<14, 1<<16};

    for (int i=0; i<g_nr_stream; i++) {
	streams[i] = g4c_alloc_stream();
	prepare_eval_item(items+i, nrpkts[sizeof(nrpkts)/sizeof(int)-1], 16, streams[0]);
    }
    g4c_stream_sync(streams[0]);

    for (int j=0; j<sizeof(nrpkts)/sizeof(int); j++) {
	timingval tv = timing_start();
	for (int i=0; i<g_nr_stream; i++) {
	    g4c_h2d_async(items[i].haddrs, items[i].daddrs,
			  nrpkts[j]*sizeof(uint32_t), streams[i]);
	    g4c_ipv4_gpu_lookup(items[i].dlpmt, items[i].daddrs,
				items[i].dports, 2, nrpkts[j], streams[i]);
	    g4c_d2h_async(items[i].dports, items[i].hports, sizeof(uint8_t)*nrpkts[j],
			  streams[i]);
	}
	for (int i=0; i<g_nr_stream; i++)
	    g4c_stream_sync(streams[i]);
	int64_t usec = timing_stop(&tv)/g_nr_streams;
	printf("GPU size %d pkts, time us %ld, rate %.3lf Mpkts/s\n",
	       nrpkts[j], usec, ((double)nrpkts[j])/(double)usec);

	timingval tvc timing_start();
	for(int i=0; i<nrpkts[j]; i++) {
	    items[i].hports[i] = g4c_ipv4_lookup(items[i].hlpmt, items[i].haddrs[i]);
	}
	int64_t cus = timing_stop(&tv2);
	printf("CPU size %d pkts, time us %ld, rate %.3lf Mpkts/s\n\n",
	       nrpkts[j], cus, ((double)nrpkts[j])/(double)cus);
    }

    g4c_exit();
    
    return 0;
}
