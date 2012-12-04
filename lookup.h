#ifndef __G4C_LOOKUP_H__
#define __G4C_LOOKUP_H__

#ifdef __cplusplus
extern "C" {
#endif

#define G4C_LPM_NULL_NODE 0

    typedef struct {
        uint32_t addr;
        uint32_t mask;
        uint8_t nnetbits;
        uint8_t port;
    } g4c_ipv4_rt_entry;

    typedef struct {
        uint8_t port;
        int children[2];
    } g4c_lpm_1b_node;

    typedef struct {
        uint8_t port;
        int children[4];
    } g4c_lpm_2b_node;

    typedef struct {
        uint8_t port;
        int children[16];    
    } g4c_lpm_4b_node;

    typedef struct {
        int nbits;
        int nnodes;
        uint8_t fport;
        union {
            g4c_lpm_1b_node b1[0];
            g4c_lpm_2b_node b2[0];
            g4c_lpm_4b_node b4[0];
        } nodes;
    } g4c_lpm_tree;

    /*
     * Some strict pre-conditions to the follow functions:
     * - pointers should never be 0, no NULL pointer checking.
     * - number of entries should never be 0, no 0 entry checking.
     * - bits should never be 0, no checking.
     * - stream handle should be valid, no validity checking.
     */
    
    uint8_t g4c_ipv4_lookup(g4c_lpm_tree *lpmtrie, uint32_t addr);
    int g4c_ipv4_gpu_lookup(g4c_lpm_tree *dlpmtrie,
                            uint32_t *daddrs, uint8_t *dports, int n, int s);
    
    g4c_lpm_tree* g4c_build_lpm_tree(g4c_ipv4_rt_entry *ents, int n,
                                     int nbits, uint8_t fport);

    
#define g4c_srt_port(entry) ((uint8_t)((entry)&0xff))
#define g4c_srt_subnet(entry) ((entry)&0xffffff00)
#define g4c_srt_subnet_idx(entry) (g4c_srt_subnet(entry)>>8)
#define g4c_srt_entry(subnet_idx, port)                 \
    ((uint32_t)(((subnet_idx)<<8) | ((port)&0xff)))
#define g4c_srt_entry2(subnet, port)            \
    (((subnet) & 0xffffff00) | ((port) &0xff))

    int g4c_build_static_routing_table(g4c_ipv4_rt_entry *ents, int n,
                                       uint8_t fport, uint32_t *srt);
    uint8_t g4c_ipv4_static_lookup(uint32_t *srt, uint32_t addr);
    int g4c_ipv4_gpu_static_lookup(uint32_t *dsrt, uint32_t *daddrs,
                                   uint8_t *dports, int n, int s);

    
#ifdef __cplusplus
}
#endif
#endif
