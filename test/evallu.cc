#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <string.h>
#include "../g4c.h"
#include "../g4c_lpm.h"
#include "utils.h"

int g_nbits = 4;
int g_nrt = 1024;

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
	x = x<0?-x:x;
	(*ents)[i].nnetbits = 8*(x+1);
	(*ents)[i].mask = (uint32_t)0xffffffff<<(32-(*ents)[i].nnetbits);
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
	(*ents)[i].port = (uint8_t)rand()%127;
    }

    return 0;	
}

static int
prepare_eval_item(rtlu_eval *item, int n, int nrt, int s)
{
    g4c_rte_t *ents = 0;
    gen_rt_entries(&ents, nrt);
    
    g4c_lpmtree_t *lpmt = g4c_build_lpm_tree(ents, nrt, g_nbits, 0);

    size_t tsz = sizeof(g4c_lpm_tree);
    switch(g_nbits) {
    case 1:
	tsz += sizeof(g4c_lpmnode1b_t)*lpmt->nnodes;
	break;
    case 2:
	tsz += sizeof(g4c_lpmnode2b_t)*lpmt->nnodes;
	break;
    case 4:
	tsz += sizeof(g4c_lpmnode4b_t)*lpmt->nnodes;
	break;
    }
    item->hlpmt = (g4c_lpmtree_t*)g4c_alloc_page_lock_mem(tsz);
    item->dlpmt = (g4c_lpmtree_t*)g4c_alloc_dev_mem(tsz);
    memcpy(item->hlpmt, lpmt, tsz);
    
    g4c_h2d_async(item->hlpmt, item->dlpmt, tsz, s);
    free(lpmt);
    free(ents);

    item->n = n;
    item->haddrs = 0;
    gen_ip_addrs(&item->haddrs, n);
    item->daddrs = (uint32_t*)g4c_alloc_dev_mem(sizeof(uint32_t)*n);
    item->hports = (uint8_t*)g4c_alloc_page_lock_mem(sizeof(uint8_t)*n);
    item->dports = (uint8_t*)g4c_alloc_dev_mem(sizeof(uint8_t)*n);

    return 0;
}


int g_nr_stream = 3;

int main(int argc, char *argv[])
{
    if (argc > 1)
	g_nrt = atoi(argv[1]);
    if (argc > 2)
	g_nbits = atoi(argv[2]);
    if (argc > 3)
	g_nr_stream = atoi(argv[3]);

    printf("G4C IP Lookup benchmakr, %d entries, %d bits table, %d streams\n",
	   g_nrt, g_nbits, g_nr_stream);
    eval_init();
    
    struct rtlu_eval *items = (struct rtlu_eval*)malloc(sizeof(struct rtlu_eval)*g_nr_stream);
    int *streams = (int*)malloc(sizeof(int)*g_nr_stream);

    int nrpkts[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

    for (int i=0; i<g_nr_stream; i++) {
	streams[i] = g4c_alloc_stream();
	prepare_eval_item(items+i, nrpkts[sizeof(nrpkts)/sizeof(int)-1], g_nrt, streams[0]);
    }
    g4c_stream_sync(streams[0]);

    {
	// warming up:
	timingval tv = timing_start();
	for (int i=0; i<g_nr_stream; i++) {
	    g4c_h2d_async(items[i].haddrs, items[i].daddrs,
			  nrpkts[0]*sizeof(uint32_t), streams[i]);
	    g4c_ipv4_gpu_lookup(items[i].dlpmt, items[i].daddrs,
				items[i].dports, g_nbits, nrpkts[0], streams[i]);
	    g4c_d2h_async(items[i].dports, items[i].hports, sizeof(uint8_t)*nrpkts[0],
			  streams[i]);
	}
	for (int i=0; i<g_nr_stream; i++)
	    g4c_stream_sync(streams[i]);
	int64_t usec = timing_stop(&tv)/g_nr_stream;
//	printf("Warming up test:\nGPU size %d pkts, time us %ld, rate %.3lf Mpkts/s\n\n",
//	       nrpkts[0], usec, ((double)nrpkts[0])/(double)usec);
    }

    int64_t *custimes = new int64_t[sizeof(nrpkts)/sizeof(int)];
    int64_t *gustimes = new int64_t[sizeof(nrpkts)/sizeof(int)];

    for (int j=0; j<sizeof(nrpkts)/sizeof(int); j++) {
	timingval tv = timing_start();
	for (int i=0; i<g_nr_stream; i++) {
	    g4c_h2d_async(items[i].haddrs, items[i].daddrs,
			  nrpkts[j]*sizeof(uint32_t), streams[i]);
	    g4c_ipv4_gpu_lookup(items[i].dlpmt, items[i].daddrs,
				items[i].dports, g_nbits, nrpkts[j], streams[i]);
	    g4c_d2h_async(items[i].dports, items[i].hports, sizeof(uint8_t)*nrpkts[j],
			  streams[i]);
	}
	for (int i=0; i<g_nr_stream; i++)
	    g4c_stream_sync(streams[i]);
	gustimes[j] = timing_stop(&tv);
	
	timingval tvc = timing_start();
	for (int b=0; b<g_nr_stream; b++) {
	    for(int i=0; i<nrpkts[j]; i++) {
		items[b].hports[i] = g4c_ipv4_lookup(items[b].hlpmt, items[b].haddrs[i]);
	    }
	}

	custimes[j] = timing_stop(&tvc);
    }

    for (int j=0; j<sizeof(nrpkts)/sizeof(int); j++) {
	printf("GPU size %5d pkts, time us %6ld, rate %8.3lf Mpkts/s\n",
	       nrpkts[j], gustimes[j]/g_nr_stream, ((double)nrpkts[j])/(double)(gustimes[j]/g_nr_stream));
    }

    for (int j=0; j<sizeof(nrpkts)/sizeof(int); j++) {
	printf("CPU size %5d pkts, time us %6ld, rate %8.3lf Mpkts/s\n",
	       nrpkts[j], custimes[j]/g_nr_stream, ((double)nrpkts[j]*g_nr_stream)/(double)(custimes[j]));
    }

    g4c_exit();
    
    return 0;
}
