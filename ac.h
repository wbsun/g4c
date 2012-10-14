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
	
#ifdef __cplusplus
}
#endif	

#endif
