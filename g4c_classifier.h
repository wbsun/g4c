#ifndef __G4C_CLASSIFIER_H__
#define __G4C_CLASSIFIER_H__

#ifdef __cplusplus
extern "C" {
#endif

#define G4C_IPA_STATE_SIZE 256

    typedef struct _g4c_pattern_t {
        uint32_t src_addr;  // low addr [3][2][1][0] high addr, 13.12.11.10 -> [10][11][12][13]
        int nr_src_netbits;
        uint32_t dst_addr;
        int nr_dst_netbits;
        int16_t src_port;   // no large port support for simplicity. negative means no such field.
        int16_t dst_port;
        int16_t proto;      // negative means no such field.
        int idx;
    } g4c_pattern_t;

    typedef struct _g4c_state_t {
        int id;
    } g4c_state_t;

    typedef int32_t g4c_cl_state_t;

    typedef struct _g4c_classifier_t {
        void *mem;
        size_t memsz;

        int nstates;
        int *transitions;
    } g4c_classifier_t;
    

#ifdef __cplusplus
}
#endif
#endif
