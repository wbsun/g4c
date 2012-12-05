#ifndef __G4C_TEST_UTILS_H__
#define __G4C_TEST_UTILS_H__
#include <stdint.h>
#include <sys/time.h>

#define interval_us(oldv, newv)					\
    ((int64_t)(1000000*((newv).tv_sec - (oldv).tv_sec)) +	\
     ((int64_t)((newv).tv_usec) - (int64_t)((oldv).tv_usec)))

typedef struct {
    struct timeval oldt;
    struct timeval newt;
} timingval;

inline static timingval
timing_start() {
    timingval tv;
    gettimeofday(&tv.oldt, 0);
    return tv;
}

inline static int64_t
timing_stop(timingval *tv) {
    gettimeofday(&tv->newt, 0);
    return interval_us(tv->oldv, tv->newv);
}

inline static int64_t
timing_elapsed(timingval *tv) {
    return interval_us(tv->oldv, tv->newv);
}

#endif
