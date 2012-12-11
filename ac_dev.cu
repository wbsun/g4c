#include <cuda.h>
#include "ac.hh"
#include "g4c.hh"

__global__ void
gpu_ac_match_general(char *strs, int stride, int *lens, unsigned int *ress,
		     ac_dev_machine_t *acm)
{
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    char *mystr = strs + (id*stride);
    unsigned int *res = ress + id*AC_ALPHABET_SIZE;

    char c;
    ac_state_t *st = acm_state(acm, 0);
    for (int i=0; i<stride; i++) {
	c = mystr[i];
	if (c>=0) {
	    int nid = acm_state_transitions(acm, st->id)[c];
	    st = acm_state(acm, nid);			
	    if (st->noutput > 0) {
		for (int j=0; j<st->noutput; j++) {
		    int ot = acm_state_output(
			acm,
			st->output)[j];
		    res[ot]++;
		}
	    }
	}
    }
}

__device__ inline void
__update_single_result(ac_state_t *st, int idx, unsigned int *res, int *output)
{
    *res += st->noutput;
}

__device__ inline void
__update_full_results(ac_state_t *st, int idx, unsigned int *res, int *output)
{
    for (int i=0; i<st->noutput; i++)
	res[output[i]]++;
}

__global__ void
gpu_ac_match(char *strings, int stride, int maxlen, int *lens,
	     unsigned int *ress, ac_dev_machine_t *acm, unsigned int match_type)
{
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    int nlen;
    char *mystr = strings + (id*stride);
    unsigned int *res;
    void (*res_handler) (ac_state_t *, int, unsigned int*, int *);
	
    switch(match_type&AC_MATCH_LEN_MASK) {
    case AC_MATCH_LEN_MAX_LEN:
	nlen = maxlen;
	break;
		
    case AC_MATCH_LEN_ALL_STRIDE:
	nlen = stride;
	break;
		
    case AC_MATCH_LEN_NORMAL:
//	case AC_MATCH_LEN_LENGTH:
    default:
	nlen = lens[id];
	break;
    }

    switch(match_type&AC_MATCH_RES_MASK) {
    case AC_MATCH_RES_SINGLE:
	res = ress + id;
	res_handler = __update_single_result;
	break;
//	case AC_MATCH_RES_NORMAL:
    case AC_MATCH_RES_FULL:
    default:
	res = ress+id*AC_ALPHABET_SIZE;
	res_handler = __update_full_results;
	break;
    }

    char c;
    ac_state_t *st = acm_state(acm, 0);
    for (int i=0; i<nlen; i++) {
	c = mystr[i];
	if (match_type & AC_MATCH_CHAR_CHECK) {
	    if (c<0)
		continue;
	}
	int nid = acm_state_transitions(acm, st->id)[c];
	st = acm_state(acm, nid);
	res_handler(st, i, res, acm_state_output(acm, st->output));		
    }
}


extern "C" size_t
ac_dev_acm_size(ac_machine_t *hacm)
{
    return g4c_ptr_offset(hacm->patterns, hacm->states);
}

extern "C" void
ac_free_dev_acm(ac_dev_machine_t **pdacm)
{
    ac_dev_machine_t *dacm = *pdacm;
    if (dacm) {
	if (dacm->dev_self)
	    g4c_free_dev_mem(dacm->dev_self);
	if (dacm->mem)
	    g4c_free_dev_mem(dacm->mem);
	g4c_free_host_mem(dacm);
	*pdacm = 0;
    }	
}

extern "C" int
ac_prepare_gmatch(ac_machine_t *hacm, ac_dev_machine_t **pdacm, int s)
{
    ac_dev_machine_t *dacm = *pdacm;
    if (!dacm) {
	*pdacm = (ac_dev_machine_t*)
	    g4c_alloc_page_lock_mem(sizeof(ac_dev_machine_t));
	if (!*pdacm) {
	    return -ENOMEM;
	}
	dacm = *pdacm;
	memset(dacm, 0, sizeof(ac_dev_machine_t));		
    }
	
    if (!dacm->dev_self) {
	dacm->dev_self = (ac_dev_machine_t*)
	    g4c_alloc_dev_mem(sizeof(ac_dev_machine_t));
	if (!dacm->dev_self) {
	    return -ENOMEM;
	}
    }

    if (!dacm->mem) {
	dacm->memsz = ac_dev_acm_size(hacm);
	dacm->mem = g4c_alloc_dev_mem(dacm->memsz);
	if (!dacm->mem)
	    return -ENOMEM;
    }

    dacm->memflags = hacm->memflags;
    dacm->nstates = hacm->nstates;
    dacm->noutputs = hacm->noutputs;

    dacm->states = (ac_state_t*)dacm->mem;
    dacm->transitions = (int*)g4c_ptr_add(
	dacm->states,
	g4c_ptr_offset(hacm->transitions,
		       hacm->states));
    dacm->outputs = (int*)g4c_ptr_add(
	dacm->states,
	g4c_ptr_offset(hacm->outputs,
		       hacm->states));

    dacm->hacm = hacm;
	
    int rt = g4c_h2d_async(
	dacm, dacm->dev_self, sizeof(ac_dev_machine_t), s);
    rt |= g4c_h2d_async(hacm->mem, dacm->mem, dacm->memsz, s);
		
    return rt;
}

extern "C" int
ac_gmatch(char *dstrs, int nstrs, int stride, int *dlens,
	  unsigned int *dress, ac_dev_machine_t *dacm, int s)
{
    cudaStream_t st = g4c_get_stream(s);
    int nblocks = g4c_round_up(nstrs/32, 32);

    gpu_ac_match_general<<<nblocks, 32, 0, st>>>(
	dstrs, stride, dlens, dress, dacm);
    return 0;
}

extern "C" int
ac_gmatch2(char *dstrs, int nstrs, int stride, int *dlens,
	   unsigned int *dress, ac_dev_machine_t *dacm, int s,
	   unsigned int mtype)
{
    cudaStream_t st = g4c_get_stream(s);
    int nblocks = g4c_round_up(nstrs/32, 32);

    return 0;
}

/*
 * May not need this.
 */
extern "C" int
ac_gmatch_finish(int nstrs, unsigned int *dress, unsigned int *hress,
		 int s)
{
    return g4c_d2h_async(dress, hress,
			 nstrs*AC_ALPHABET_SIZE*sizeof(unsigned int),
			 s);
}
