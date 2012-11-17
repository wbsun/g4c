#ifndef __G4C_AC_H__
#define __G4C_AC_H__

#ifdef __cplusplus
extern "C" {
#endif

#define AC_ALPHABET_SIZE 128
	
	typedef struct _ac_state_t {
		int id;
		int prev;
		int *output;
		int noutput;
	} ac_state_t;

#define acm_state_transitions(pacm, sid) ((pacm)->transitions +\
					 (sid)*AC_ALPHABET_SIZE)
#define acm_state(pacm, sid) ((pacm)->states + (sid))
#define acm_pattern(pacm, pid) (*((pacm)->patterns + (pid)))

	typedef struct _ac_machine_t {
		void *mem;
		unsigned long memsz;

// Ignore flags for now 
#define ACM_PATTERN_PTRS_INSIDE     0x00000001
#define ACM_PATTERNS_INSIDE         0x00000002
#define ACM_OWN_PATTERN_PTRS        0x00000004
#define ACM_OWN_PATTERNS            0x00000008 
#define ACM_BUILD_COPY_PATTERN_PTRS 0x00000010
#define ACM_BUILD_COPY_PATTERNS     0x00000020
		unsigned int memflags;

		ac_state_t *states;
		int nstates;

		int *transitions;
		int *outputs;
		int noutputs;

		char **patterns;
		int npatterns;		
	} ac_machine_t;

	int ac_build_machine(
		ac_machine_t *acm,
		char **patterns,
		int npatterns,
		unsigned int memflags);
	void ac_release_machine(ac_machine_t *acm);

	
#define ac_res_found(r) ((r)>>31)
#define ac_res_location(r) ((r)&0x7fffffff)
#define ac_res_set_found(r, f) ((r)|(f<<31))
#define ac_res_set_location(r, loc) (((r)&0x80000000)|loc)
	
	/*
	 * Match pattern in str to at most len characters with acm.
	 *
	 * If res is not NULL, it must be unsigned int[nr_patterns], and results
	 *   are recorded in it. For each pattern i, if matched, res[i]:[31] is 1,
	 *   and res[i]:[30-0] is the location; otherwise res[i] is untouched. So
	 *   res should be memset-ed to 0 by caller before calling.
	 *
	 * once: return at first match, don't match all chars in str.	 
	 *
	 * Reture: # of matches, or -1 on error.
	 *
	 */
	int ac_match(char *str, int len, unsigned int *res, int once, ac_machine_t *acm);

	/*
	 * Prepare ACM matching on GPU by copying ACM in host memory to
	 * device memory.
	 *
	 * hacm is the ACM machine in host memory. dacm is the device one.
	 *
	 * If dacm is NULL, device memory for dacm is allocated according to
	 *   hacm.
	 * If dacm is not NULL, the caller must ensure it has enough space to
	 *   hold a copy of hacm.
	 *
	 * Reture: 1 on success, 0 otherwise.
	 *
	 */
	int ac_prepare_gmatch(ac_machine_t *hacm, ac_machine_t **dacm, int s);

	int ac_gmatch(char *dstrs, int nstrs, int stride, int *dlens,
		      unsigned int *dress, ac_machine_t *dacm, int s);
	int ac_gmatch_finish(int nstrs, unsigned int *dress, unsigned int *hdress,
			     int s);
	
#ifdef __cplusplus
}
#endif	

#endif
