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


int __ac_gmatch(char *dstrs, int nstrs, int stride, int *dlens,
		unsigned int *dress, ac_dev_machine_t *dacm, int s)
{
	cudaStream_t st = g4c_get_stream(s);
	int nblocks = g4c_round_up(nstrs/32, 32);

	gpu_ac_match_general<<<nblocks, 32, 0, st>>>(
		dstrs, stride, dlens, dress, dacm);
	return 0;
}
