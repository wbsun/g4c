#include <cuda.h>
#include "ac.hh"
#include "g4c.hh"

__global__ void
gpu_ac_match(char *strs, int stride, int *lens, unsigned int *ress,
	     ac_dev_machine_t *acm)
{
	int id = blockIdx.x*blockDim.x + threadIdx.x;
	char *mystr = strs + (id*stride);
	int len = lens[id];
	unsigned int *res = ress + id*AC_ALPHABET_SIZE;

	char c;
	ac_state_t *st = acm_state(acm, 0);
	for (int i=0; i<stride; i++) {
		//for (int i=0; i<len; i++) {
		c = str[i];
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


int __ac_gmatch(char *dstrs, int nstrs, int stride, int *dlens,
		unsigned int *dress, ac_dev_machine_t *dacm, int s)
{
	cudaStream_t st = g4c_get_stream(s);
	int nblocks = g4c_round_up(nstrs/32, 32);

	gpu_ac_match<<<nblocks, 32, 0, st>>>(
		dstrs, stride, dlens, dress, dacm);
	return 0;
}
