/*
 * common.h — Shared packet structure between device (inject_funcs.cu)
 *             and host (mem_reuse_tracker.cu).
 */
#pragma once
#include <stdint.h>

/* One packet per warp-first-lane global load access.
 * 128-byte L2 cache-line index (addr >> 7) + launch ID for grouping. */
typedef struct {
    uint64_t cacheline_index; /* addr >> 7 */
    uint64_t grid_launch_id;  /* which kernel invocation */
} cl_access_t;
