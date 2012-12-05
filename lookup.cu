#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include "lookup.h"
#include "g4c.hh"

template<int BITS>
class trie_node {
public:
    int id;
    uint8_t port;
    int bits;
    trie_node<BITS> *children[(1<<BITS)];
    trie_node<BITS> *prev;
    trie_node<BITS> *next;
    
    trie_node():bits(BITS), prev(0), next(0) {
	for (int i=0; i<(1<<BITS); i++)
	    children[i] = 0;	
    }
};

template<int BITS>
class node_store {
public:
    int id_seq;
    int bits;
    trie_node<BITS> head;
    node_store() : id_seq(0),
		   bits(BITS) {
	head.prev = &head;
	head.next = &head;
    }

    void append(trie_node<BITS> *n) {
	n->next = &head;
	n->prev = head.prev;
	head.prev->next = n;
	head.prev = n;
    }

    template<class node_type> g4c_lpm_tree *
    build_lpm_tree(g4c_ipv4_rt_entry *ents, int n, uint8_t fport,
		   node_type dummy) {
	trie_node<BITS> *root = new trie_node<BITS>();
	root->id = id_seq++;
	root->port = fport;
	append(root);

	trie_node<BITS> *node = root;
	for (int i=0; i<n; i++) {
	    uint32_t val = 0;
	    int ite;
	    for_bits_h2l(32-BITS, 32-ents[i].nnetbits,
			 ents[i].addr, val, ite, BITS) {
		if (!node->children[val]) {
		    trie_node<BITS> *p = new trie_node<BITS>();
		    p->id = id_seq++;
		    p->port = fport;
		    append(p);
		    node->children[val] = p;
		}
		node = node->children[val];
	    }
	    node->port = ents[i].port;
	}   

	g4c_lpm_tree * lpmt = (g4c_lpm_tree*)malloc(
	    sizeof(g4c_lpm_tree) + id_seq*sizeof(node_type));
	if (!lpmt)
	    return 0;

	lpmt->nbits = BITS;
	lpmt->nnodes = id_seq;
	lpmt->fport = fport;
	node = root;
	node_type *tnds = (node_type*)lpmt->nodes.b1;
	for (int i=0; i<lpmt->nnodes; i++) {
	    if (node == 0) { // report problem
		free(lpmt);
		return 0;
	    }
	
	    tnds[i].port = node->port;
	    for (int j=0; j<(1<<BITS); j++) {
		if (node->children[j])
		    tnds[i].children[j] = node->children[j]->id;
		else
		    tnds[i].children[j] = 0;
	    }
	    node = node->next;	
	}

	return lpmt;
    }

    void clear() {
	trie_node<BITS> *node = head.next;
	while (node != &head) {
	    node = node->next;
	    delete node->prev;
	}
	head.next = head.prev = 0;
    }    
};

template<class node_type, int BITS> static g4c_lpm_tree *
__build_lpm_tree(node_store<BITS> store,
		 g4c_ipv4_rt_entry *ents, int n, uint8_t fport,
		 node_type dummy)
{
    g4c_lpm_tree * lpmt = store.build_lpm_tree(ents, n, fport, dummy);
    store.clear();
    return lpmt;
}

extern "C" g4c_lpm_tree *
g4c_build_lpm_tree(g4c_ipv4_rt_entry *ents, int n, int nbits, uint8_t fport)
{    
    switch(nbits) {
    case 1:
	return __build_lpm_tree(
	    node_store<1>(), ents, n, fport, g4c_lpm_1b_node());
    case 2:
	return __build_lpm_tree(
	    node_store<2>(), ents, n, fport, g4c_lpm_2b_node());
    case 4:
	return __build_lpm_tree(
	    node_store<4>(), ents, n, fport, g4c_lpm_4b_node());
    default:
	return 0;
    }
}

template<class node_type> uint8_t
__ipv4_lookup(g4c_lpm_tree *lpmtrie, uint32_t addr, node_type dummy)
{
    uint32_t val;
    int ite;

    node_type *node = (node_type*)lpmtrie->nodes.b1;
    int nid = 0;
    uint8_t port = lpmtrie->fport;
    for_bits_h2l(32-lpmtrie->nbits, 0, addr, val, ite, lpmtrie->nbits) {
	if (node[nid].port != lpmtrie->fport)
	    port = node[nid].port;
	
	if (node[nid].children[val]) {
	    nid = node[nid].children[val];
	} else
	    break;
    }

    return port;
}

extern "C" uint8_t
g4c_ipv4_lookup(g4c_lpm_tree *lpmtrie, uint32_t addr)
{
    switch(lpmtrie->nbits) {
    case 1:
	return __ipv4_lookup(lpmtrie, addr, g4c_lpm_1b_node());
    case 2:
	return __ipv4_lookup(lpmtrie, addr, g4c_lpm_2b_node());
    case 4:
	return __ipv4_lookup(lpmtrie, addr, g4c_lpm_4b_node());
    default:
	return lpmtrie->fport;
    }
}


template <typename node_type> __global__ void
gpu_lpm_lookup(g4c_lpm_tree *lpmt,
	       uint32_t *addrs,
	       uint8_t *ports,
	       int n, node_type *dummy)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    uint32_t val, addr = addrs[id];
    int ite, nid=0;
    node_type *node = (node_type*)lpmt->nodes.b1;
    for_bits_h2l(32-lpmt->nbits, 0, addr, val, ite, lpmt->nbits) {
	if (node[nid].children[val]) {
	    nid = node[nid].children[val];
	} else {
	    ports[id] = node[nid].port;
	    break;
	}
    }
}


extern "C" int
g4c_ipv4_gpu_lookup(g4c_lpm_tree *dlpmt,
		    uint32_t *daddrs,
		    uint8_t *dports,
		    int nbits, int n, int s)
{
    cudaStream_t stream = g4c_get_stream(s);

    switch(nbits) {
    case 1:
	gpu_lpm_lookup<<<n/32, 32, 0, stream>>>(
	    dlpmt, daddrs, dports, n, (g4c_lpm_1b_node*)(0));
	break;
    case 2:
	gpu_lpm_lookup<<<n/32, 32, 0, stream>>>(
	    dlpmt, daddrs, dports, n, (g4c_lpm_2b_node*)(0));
	break;
    case 4:
	gpu_lpm_lookup<<<n/32, 32, 0, stream>>>(
	    dlpmt, daddrs, dports, n, (g4c_lpm_4b_node*)(0));
	break;
    default:
	gpu_lpm_lookup<<<n/32, 32, 0, stream>>>(
	    dlpmt, daddrs, dports, n, (g4c_lpm_1b_node*)(0));
	break;
    }

    return 0;
}


// Import as soon as we need
#include <vector>
#include <algorithm>
using namespace std;

static bool
__ipv4_rt_ent_less(g4c_ipv4_rt_entry *e1, g4c_ipv4_rt_entry *e2)
{
    return e1->addr < e2->addr;
}

extern "C" int
g4c_build_static_routing_table(g4c_ipv4_rt_entry *ents, int n,
			       uint8_t fport, uint32_t *srt)
{
    vector<g4c_ipv4_rt_entry *> vents;
    g4c_ipv4_rt_entry first, last;

    first.addr = 0;
    first.port = fport;
    last.addr = (uint32_t)(~0);
    last.port = fport;
    
    vents.reserve(n+2);
    vents.push_back(&first);
    
    for (int i=0; i<n; i++)
	vents.push_back(ents+i);
    sort(vents.begin()+1, vents.end(), __ipv4_rt_ent_less);
    vents.push_back(&last);

    vector<g4c_ipv4_rt_entry *>::iterator n1 = vents.begin();
    vector<g4c_ipv4_rt_entry *>::iterator n2 = n1+1;
    do {
	for (int i = ((*n1)->addr>>8);
	     i < ((*n2)->addr>>8); i++)
	    srt[i] = g4c_srt_entry2((*n1)->addr, (*n1)->port);
	++n1;
	++n2;	
    } while (n1 != vents.end());

    return 0;
}

extern "C" uint8_t
g4c_ipv4_static_lookup(uint32_t *srt, uint32_t addr)
{
    return g4c_srt_port(srt[g4c_srt_subnet_idx(addr)]);
}

__global__ void
gpu_static_lookup(uint32_t *srt, uint32_t *addrs, uint8_t *ports, int n)
{
}

extern "C" int
g4c_ipv4_gpu_static_lookup(uint32_t *dsrt, uint32_t *daddrs,
			   uint8_t *dports, int n, int s)
{
    cudaStream_t stream = g4c_get_stream(s);
    gpu_static_lookup<<<n/32, 32, 0, stream>>>(dsrt, daddrs, dports, n);
    return 0;
}
