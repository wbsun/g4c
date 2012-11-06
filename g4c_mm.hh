#ifndef __G4C_MM_HH__
#define __G4C_MM_HH__

#include "g4c.h"

// only for x86_64
typedef unsigned char u8_t;
typedef unsigned short u16_t;
typedef unsigned int u32_t;
typedef unsigned long u64_t;

struct MemAllocInfo {
	u32_t start_unit;
	u32_t order;
	MemAllocInfo(u32_t su, u32_t od): start_unit(su), order(od) {};
	MemAllocInfo(const MemAllocInfo &v) {
		start_unit = v.start_unit;
		order = v.order;
	}
	~MemAllocInfo() {}
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
	MMContext() {}
	MMContext(const MMContext& v) {
		index = v.index;
		
		base_addr = v.base_addr;
		size = v.size;
		
		unit_size = v.unit_size;
		unit_shift = v.unit_shift;
		unit_mask = v.unit_mask;
		nr_units = v.nr_units;

		nr_orders = v.nr_orders;
		order_begin = v.order_begin;
		order_end = v.order_end;

		allocated_chunks = v.allocated_chunks;
		free_chunks = v.free_chunks;
	}
	~MMContext() {}
};

int create_mm_context(u64_t base_addr, u64_t size, u32_t unit_order);
u64_t alloc_region(int mmc_idx, u64_t size);
bool free_region(int mmc_idx, u64_t addr);
bool release_mm_context(int hdl);

#endif
