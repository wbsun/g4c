#ifndef __G4C_CLASSIFIER_H__
#define __G4C_CLASSIFIER_H__

#ifdef __cplusplus
extern "C" {
#endif

#define G4C_IPA_STATE_SIZE 256
#define G4C_IPA_STATE_BITS 8

#ifndef PORT_BITS
#define PORT_BITS 16
#endif
#define PORT_STATE_SIZE (1<<PORT_BITS)
#define PORT_MASK (((uint32_t)0xffff)>>(16-PORT_BITS))
#define get_eport(_p) ((int)((_p)&PORT_MASK))

#ifndef PROTO_BITS
#define PROTO_BITS 5
#endif
#define PROTO_STATE_SIZE (1<<PROTO_BITS)
#define PROTO_MASK (((uint16_t)0xffff)>>(16-PROTO_BITS))
#define get_eproto(_p) ((int)((_p)&PROTO_MASK))

    typedef g4c_cl_sid_t int;

    // All fields in network order
    typedef struct _g4c_pattern_t {
        uint32_t src_addr; 
        int nr_src_netbits;
        uint32_t dst_addr;
        int nr_dst_netbits;
        int32_t src_port;   // negative means no such field. lower two bytes in network order
        int32_t dst_port;
        int16_t proto;      // negative means no such field.
        int idx;
    } g4c_pattern_t;

#define G4C_CL_RES_SZ_ALIGN 4
#define cl_ipa_trans_tbl(bp, id) ((bp)+((id)<<G4C_IPA_STATE_BITS))
#define cl_ipa_trans(bp, id, v) (cl_ipa_trans_tbl((bp), (id))+(v))
#define cl_res(bp, id, stride) ((bp)+(id)*(stride))
#define cl_trans_tbl(bp, id, stsz) ((bp)+((id)*stsz))
    
    typedef struct _g4c_classifier_t {
        void *mem;
        void *devmem;
        size_t memsz;
        int nrules;
        int res_sz;
        uint32_t res_stride;

        int nr_saddr_sts;
        int *saddr_trs;
        uitn32_t *saddr_ress;

        int nr_daddr_sts;
        int *daddr_trs;
        uint32_t *daddr_ress;

        int nr_sp_sts;
        int *sp_trs;
        uint32_t *sp_ress;

        int nr_dp_sts;
        int *dp_trs;
        uint32_t *dp_ress;

        int nr_pt_sts;
        int *pt_trs;
        uint32_t *pt_ress;

        int *dev_saddr_trs;
        int *dev_daddr_trs;
        int *dev_sp_trs;
        int *dev_dp_trs;
        int *dev_pt_trs;
        uint32_t *dev_saddr_ress;
        uint32_t *dev_daddr_ress;
        uint32_t *dev_sp_ress;
        uint32_t *dev_dp_ress;
        uint32_t *dev_pt_ress;        
    } g4c_classifier_t;

    g4c_classifier_t *g4c_create_classifier(g4c_pattern_t *ptn, int nptn, int create_dev, int stream);
    int g4c_cpu_classify_pkt(g4c_classifier_t *gcl, uint8_t *ttlptr);

    // res_stride is for int*, not byte ptr, res_ofs the same.
    int g4c_gpu_classify_pkts(g4c_classifier_t *dgcl, int npkts,
                              uint8_t *data, uint32_t stride, uint32_t ttl_ofs, uint32_t l3hdr_ofs,
                              int *ress, uint32_t res_stride, uint32_t res_ofs,
                              int s);

#ifdef __cplusplus
}
#endif
#endif
