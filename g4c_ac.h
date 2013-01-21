#ifndef __G4C_AC_H__
#define __G4C_AC_H__

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AC_MATCH_LEN_MASK        0x0000000f
#define AC_MATCH_LEN_NORMAL      0x0
#define AC_MATCH_LEN_LENGTH      0x0
#define AC_MATCH_LEN_ALL_STRIDE  0x00000001
#define AC_MATCH_LEN_MAX_LEN     0x00000002

#define AC_MATCH_RES_MASK        0x000000f0
#define AC_MATCH_RES_NORMAL      0x0
#define AC_MATCH_RES_FULL        0x0
#define AC_MATCH_RES_SINGLE      0x00000010

#define AC_MATCH_CHAR_CHECK_MASK 0x80000000
#define AC_MATCH_CHAR_CHECK      0x80000000


#define AC_ALPHABET_SIZE 256
	
    typedef struct _ac_state_t {
        int id;
        int prev;
        int output;
        int noutput;
    } ac_state_t;

#define acm_state_transitions(pacm, sid) ((pacm)->transitions +         \
                                          (sid)*AC_ALPHABET_SIZE)
#define acm_state(pacm, sid) ((pacm)->states + (sid))
#define acm_pattern(pacm, pid) (*((pacm)->patterns + (pid)))
#define acm_state_output(pacm, ofs) ((pacm)->outputs + (ofs))

    typedef struct _ac_machine_t {
        void *mem;
        size_t memsz;
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

    typedef struct _ac_dev_machine_t {
        void *mem;
        size_t memsz;
        unsigned int memflags;
        ac_state_t *states;
        int nstates;
        int *transitions;
        int *outputs;
        int noutputs;
        struct _ac_dev_machine_t *dev_self;
        ac_machine_t *hacm;
    } ac_dev_machine_t;

    typedef struct _g4c_acm_t {
	void *mem;
	size_t memsz;
	void *devmem;

	int nstates;
	int *transitions;
	int *outputs;

	int *dtransitions;
	int *doutputs;
    } g4c_acm_t;

#define g4c_acm_htransitions(acm, id) \
    ((int*)((acm)->transitions + (id)*AC_ALPHABET_SIZE))
#define g4c_acm_dtransitions(acm, id) \
    ((int*)((acm)->dtransitions + (id)*AC_ALPHABET_SIZE))
#define g4c_acm_houtput(acm, id) \
    ((int*)((acm)->outputs + (id)))
#define g4c_acm_doutput(acm, id) \
    ((int*)((acm)->doutputs + (id)))

    g4c_acm_t *g4c_create_matcher(
	char **ptns, int nptns, int withdev, int stream);

    // Result values: 0 no match; pos-num, rule idx + 1.
    int g4c_gpu_acm_match(
	g4c_acm_t *dacm, int nr,
	uint8_t *ddata, uint32_t data_stride, uint32_t data_ofs,
	int *dlens,
	int *dress, uint32_t res_stride, uint32_t res_ofs,
	int s, int mtype);
    int g4c_cpu_acm_match(
	g4c_acm_t *acm, uint8_t *data, int len);
	

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
    int ac_match(char *str, int len, unsigned int *res, int once,
                 ac_machine_t *acm);

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
    int ac_prepare_gmatch(ac_machine_t *hacm, ac_dev_machine_t **dacm, int s);

    size_t ac_dev_acm_size(ac_machine_t *hacm);
    void ac_free_dev_acm(ac_dev_machine_t **pdacm);

    /*
     * Caller should take care of memset dress.
     */
    int ac_gmatch(char *dstrs, int nstrs, int stride, int *dlens,
                  unsigned int *dress, ac_dev_machine_t *dacm, int s);

    int ac_gmatch2(char *dstrs, int n, int stride, int *dlens, unsigned int *dress,
                   ac_dev_machine_t *dacm, int s, unsigned int mtype);

    // res_stride and res_ofs are for int type, not bytes offset or size
    // This is a much since GPU can't read non-aligned data pointers.
    int ac_gmatch2_ofs(char *dstrs, int n, int stride, int *dlens, int *dress,
		       ac_dev_machine_t *dacm, int s, unsigned int mtype,
		       uint32_t pkt_ofs, uint32_t res_stride, uint32_t res_ofs);
    

    /*
     * Seems unneeded.
     */	
    int ac_gmatch_finish(int nstrs, unsigned int *dress, unsigned int *hress,
                         int s);
	
#ifdef __cplusplus
}
#endif	

#endif
