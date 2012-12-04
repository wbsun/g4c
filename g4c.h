#ifndef __G4C_H__
#define __G4C_H__

#ifdef __cplusplus
extern "C" {
#endif
	
#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define g4c_round_up(x, y) ((((x)-1) | __round_mask(x, y))+1)
#define g4c_round_down(x, y) ((x) & ~__round_mask(x, y))

#define g4c_ptr_add(ptr, offset) ((void*)(((unsigned char*)(ptr)) + (offset)))
#define g4c_ptr_offset(hptr, lptr) ((unsigned long)((unsigned char*)(hptr) - (unsigned char*)(lptr)))

// A hack to make sure a variable is really read from or written to memory.
#define g4c_to_volatile(x) (*((volatile __typeof(x) *)(&(x))))
#define g4c_to_ul(x) ((unsigned long)(x))

// Get most significant 1 bit, between 63 and 0.
// x is non-zero.
#define g4c_msb_u64(x) (63 - __builtin_clzl(x))

#define g4c_var_barrier(v)

#define for_bits_h2l(begin, end, data, val, ite, step)			\
    for (ite = (begin),							\
	     val = ((data)>>ite)&&(~((~((typeof(data)) 0x0))<<(step)));	\
	 ite >= (end);							\
	 ite -= (step),							\
	     val = ((data)>>ite)&&(~((~((typeof(data)) 0x0))<<(step))))    

#define G4C_PAGE_SIZE 4096
#define G4C_PAGE_SHIFT 12
#define G4C_PAGE_MASK ((unsigned long)((~(unsigned long)(0x0))<<(G4C_PAGE_SHIFT)))
	
#define G4C_MEM_ALIGN 32

#define G4C_EMCPY -10000
#define G4C_EKERNEL -10001
#define G4C_EMM -10002

#define G4C_DEFAULT_NR_STREAMS 32
#define G4C_DEFAULT_MEM_SIZE (0x1<<30)

    /*
     * Return value: 0 means OK.
     */

    int g4c_init(int nr_streams,
                 size_t hostmem_sz,
                 size_t devmem_sz);
    void g4c_exit(void);
    void g4c_abort(void);

    void *g4c_alloc_page_lock_mem(size_t sz);
    void g4c_free_page_lock_mem(void* p);
    void *g4c_alloc_dev_mem(size_t sz);
    void g4c_free_dev_mem(void* p);

    int g4c_alloc_stream();
    void g4c_free_stream(int s);

    int g4c_stream_sync(int s);
    int g4c_stream_done(int s);

    int g4c_h2d_async(void *h, void *d, size_t sz, int s);
    int g4c_d2h_async(void *d, void *h, size_t sz, int s);

    int g4c_dev_memset(void *d, int val, size_t sz, int s);

    const char *g4c_strerror(int err);

#ifdef __cplusplus
}
#endif	

#endif
