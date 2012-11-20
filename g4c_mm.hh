#ifndef __G4C_MM_HH__
#define __G4C_MM_HH__

#include "g4c.h"

struct MemAllocInfo {
	uint32_t start_unit;
	uint32_t order;
	MemAllocInfo(uint32_t su, uint32_t od): start_unit(su), order(od) {};
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
	
	uint64_t base_addr;
	uint64_t size;

	uint32_t unit_size;
	uint32_t unit_shift; // unit_order
	uint64_t unit_mask;	
	uint32_t nr_units;
	
	uint32_t nr_orders;
	uint32_t order_begin;
	uint32_t order_end;
	
	set<MemAllocInfo, MemAllocInfoComp> allocated_chunks;
	vector<set<uint32_t> > free_chunks;
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

int create_mm_context(uint64_t base_addr, uint64_t size, uint32_t unit_order);
uint64_t alloc_region(int mmc_idx, uint64_t size);
bool free_region(int mmc_idx, uint64_t addr);
bool release_mm_context(int hdl);

#endif
