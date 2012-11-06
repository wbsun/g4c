#ifndef __G4C_MM_H__
#define __G4C_MM_H__

#ifdef __cplusplus
extern "C" {
#endif

	int g4c_new_mm_handle(void *base_addr,
			      size_t total_size,
			      unsigned int unit_order);
	void *g4c_alloc_mem(int mm_handle, size_t size);
	int g4c_free_mem(int mm_handle, void *addr);
	int g4c_release_mm_handle(int mm_handle);

#ifdef __cplusplus
}
#endif

#endif
