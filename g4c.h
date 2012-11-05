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

#define g4c_var_barrier(v) 

#define G4C_PAGE_SIZE 4096
#define G4C_PAGE_SHIFT 12
#define G4C_PAGE_MASK ((unsigned long)((~(unsigned long)(0x0))<<(G4C_PAGE_SHIFT)))
	
#define G4C_MEM_ALIGN 32

	typedef struct {
		int stream;
	} g4c_async_t;

#define G4C_EMCPY -10000
#define G4C_EKERNEL -10001

	/*
	 * Return value: 0 means OK.
	 */

	int g4c_init(void);
	void g4c_exit(void);

	void *g4c_alloc_page_lock_mem(size_t sz);
	void g4c_free_page_lock_mem(void* p, size_t sz);
	void *g4c_alloc_dev_mem(size_t sz);
	void g4c_free_dev_mem(void* p, size_t sz);

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
