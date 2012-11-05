#include <set>
#include <vector>

using namespace std;

#include "g4c_mm.hh"

vector<MMContext> mmcontexts;

void lock_mmcontexts()
{
	;
}

void unlock_mmcontexts()
{
	;
}

/**
 * base_addr must be 2^unit_order aligned;
 * size must be able to be divided by 2^unit_oder;
 *
 */
int create_mm_context(u64_t base_addr, u64_t size, u32_t unit_order)
{
	lock_mmcontexts();
	mmcontexts.push_back(MMContext());
	MMContext &mmc = mmcontexts.back();
	mmc.index = mmcontexts.size()-1;
	unlock_mmcontexts();

	mmc.base_addr = base_addr;
	mmc.size = size;

	mmc.unit_size = 0x1 << unit_order;
	mmc.unit_shift = unit_order;
	mmc.unit_mask = (u64_t)(0xffffffffffffffff)<<mmc.unit_shift;
	mmc.nr_units = (u32_t)(size >> unit_order);

	mmc.order_begin = unit_order;
	mmc.order_end = g4c_msb_u64(size);
	mmc.nr_orders = mmc.order_end - mmc.order_begin +1;

	u32_t chunk_idx = 0;
	for (u32_t i=mmc.order_begin; i<=mmc.order_end; i++) {
		mmc.free_chunks.push_back(set<u32_t>());		
		set<u32_t> & free_set = mmc.free_chunks.back();
		u64_t sz = ((u64_t)0x1)<<i;

		if (sz & mmc.size != 0) {
			free_set.insert(chunk_idx);
			chunk_idx += (sz>>unit_order);
		}
	}

	return mmc.index;
}


u64_t alloc_region(int mmc_idx, u64_t size)
{
}

bool free_region(int mmc_idx, u64_t addr)
{
}



