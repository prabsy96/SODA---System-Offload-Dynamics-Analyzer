/*
 * inject_funcs.cu — Device-side instrumentation function.
 *
 * Compiled with:
 *   nvcc -Xptxas -astoolspatch --keep-device-functions -arch=$(ARCH)
 *        -Xcompiler -fPIC -c inject_funcs.cu -o inject_funcs.o
 *
 * This function is injected before every LD.GLOBAL instruction in
 * tagged expert kernels.  Only the first active lane per warp sends
 * data to avoid duplicate cache-line entries from the same warp.
 */
#include <stdint.h>
#include "utils/utils.h"
#include "utils/channel.hpp"
#include "common.h"

extern "C" __device__ __noinline__ void
instrument_cacheline(int pred,
                     uint64_t addr,
                     uint64_t grid_launch_id,
                     uint64_t pchannel_dev) {
    /* predicated-off threads contribute nothing */
    if (!pred) return;

    /* warp-level deduplication: only the first active lane pushes */
    const int active_mask  = __ballot_sync(__activemask(), 1);
    const int laneid       = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    if (first_laneid == laneid) {
        cl_access_t acc;
        acc.cacheline_index = addr >> 7;   /* 128-byte L2 sector */
        acc.grid_launch_id  = grid_launch_id;

        ChannelDev* ch = (ChannelDev*)pchannel_dev;
        ch->push(&acc, sizeof(cl_access_t));
    }
}
