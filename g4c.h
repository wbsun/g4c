#ifndef __G4C_H__
#define __G4C_H__

#ifdef __cplusplus
extern "C" {
#endif
	
#define __round_mask(x, y) ((__typeof__(x))((y)-1))
#define g4c_round_up(x, y) ((((x)-1) | __round_mask(x, y))+1)
#define g4c_round_down(x, y) ((x) & ~__round_mask(x, y))

	typedef struct {
		int stream;
	} g4c_async_t;

#define G4C_EMCPY -10000
#define G4C_EKERNEL -10001

	int g4c_init(void);
	void g4c_exit(void);

	void* g4c_malloc(size_t sz);
	void g4c_free(void *p);

	int g4c_do_stuff_sync(void *in, void *out, int n);
	int g4c_do_stuff_async(void *in, void *out, int n, g4c_async_t *asyncdata);

	int g4c_check_async_done(g4c_async_t *asyncdata);

	const char *g4c_strerror(int err);

#ifdef __cplusplus
}
#endif	

#endif
