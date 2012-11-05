#ifndef __G4C_MM_HH__
#define __G4C_MM_HH__

#include "g4c.h"

// only for x86_64
typedef unsigned char u8_t;
typedef unsigned short u16_t;
typedef unsigned int u32_t;
typedef unsigned long u64_t;

// Get most significant bit, between 63 and 0.
#define g4c_msb_u64(x)

struct MemAllocInfo {
	u32_t start_unit;
	u32_t order;
};

struct MemAllocInfoComp {
	inline bool operator() (const MemAllocInfo &a,
				const MemAllocInfo &b) const
		{
			return a.start_unit < b.start_unit;
		}
};

struct MMContext {
	int index;
	
	u64_t base_addr;
	u64_t size;

	u32_t unit_size;
	u32_t unit_shift; // unit_order
	u64_t unit_mask;	
	u32_t nr_units;
	
	u32_t nr_orders;
	u32_t order_begin;
	u32_t order_end;
	
	set<MemAllocInfo, MemAllocInfoComp> allocated_chunks;
	vector<set<u32_t> > free_chunks;
};

#endif
