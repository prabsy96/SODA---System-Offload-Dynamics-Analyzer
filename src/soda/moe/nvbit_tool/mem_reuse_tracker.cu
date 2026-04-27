/*
 * mem_reuse_tracker.cu — NVBit in-process L2 cache-line reuse tracker
 *
 * Instruments tagged expert kernels (shared_expert, routed_expert, gate)
 * during a full model.generate() run.  Emits one JSON record per kernel
 * invocation to $SODA_NVBIT_LOG.
 *
 * Build (requires NVBit 1.7.7+ and nvcc):
 *   make NVBIT_PATH=/path/to/nvbit_release_x86_64/core
 *
 * Usage:
 *   export SODA_NVBIT_LOG=/tmp/nvbit_reuse.jsonl
 *   export SODA_EXPERT_MAP=/tmp/expert_map.json
 *   LD_PRELOAD=./mem_reuse_tracker.so python -c "model.generate(...)"
 *
 * Output (JSON lines, one per tagged kernel invocation):
 *   {"kernel_name":"cutlass_gemm_...","expert_type":"shared_expert",
 *    "invocation":3,"global_load_count":1048576,
 *    "cacheline_set_size":65536,"cacheline_hashes":[1234,...]}
 */

#include <assert.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/* every tool needs to include this once */
#include "nvbit_tool.h"
/* nvbit interface */
#include "nvbit.h"
/* device↔host ring buffer */
#include "utils/channel.hpp"
/* shared packet struct */
#include "common.h"

/* flush_channel fatbin embedded by bin2c during build */
#include "tool_func/flush_channel.c"

/* -----------------------------------------------------------------------
 * Constants
 * --------------------------------------------------------------------- */
#define CHANNEL_SIZE (1l << 20)
/* Max cache-line hashes written to JSON per invocation */
#define MAX_CL_HASHES_JSON 4096

/* -----------------------------------------------------------------------
 * Per-CUDA-context state
 * --------------------------------------------------------------------- */
enum class RecvThreadState { INIT, WORKING, STOP, FINISHED };

struct CTXstate {
    int id;
    ChannelDev*  channel_dev;
    ChannelHost  channel_host;
    CUmodule     tool_module;
    CUfunction   flush_channel_func;
    volatile RecvThreadState recv_thread_done = RecvThreadState::INIT;
    bool need_sync = false;
};

/* -----------------------------------------------------------------------
 * Global state
 * --------------------------------------------------------------------- */
/* Expert map: kernel_name → expert_type (loaded from SODA_EXPERT_MAP) */
static std::unordered_map<std::string, std::string> g_expert_map;

/* Output log */
static FILE* g_log_fp = nullptr;

/* Context state map */
static std::unordered_map<CUcontext, CTXstate*> ctx_state_map;

/* Instrumented functions */
static std::unordered_set<CUfunction> already_instrumented;

/* Per-launch accumulator: grid_launch_id → {name, type, cacheline set} */
struct InvocData {
    std::string kernel_name;
    std::string expert_type;
    uint64_t    global_load_count = 0;
    std::unordered_set<uint64_t> cacheline_set;
};
static std::unordered_map<uint64_t, InvocData> g_invoc_map;

/* Monotonic launch counter */
static uint64_t g_grid_launch_id = 0;

/* Which grid_launch_id maps to which CUfunction (for expert_type lookup) */
static std::unordered_map<uint64_t, std::string> g_launch_expert_type;
static std::unordered_map<uint64_t, std::string> g_launch_kernel_name;

/* Re-entry guard (prevent nvbit callbacks from recursing) */
static bool skip_callback_flag = false;

/* Mutexes */
static pthread_mutex_t mutex;
static pthread_mutex_t cuda_event_mutex;

/* -----------------------------------------------------------------------
 * Expert map loader
 *
 * Minimal hand-rolled parser for {"key": "value", ...} JSON.
 * --------------------------------------------------------------------- */
static void load_expert_map(const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr,
                "[SODA/NVBit] Error: cannot open SODA_EXPERT_MAP: %s\n", path);
        exit(1);
    }

    char line[8192];
    /* Read entire file into buffer */
    std::string content;
    while (fgets(line, sizeof(line), fp)) content += line;
    fclose(fp);

    /* Extract "key": "value" pairs */
    const char* p = content.c_str();
    while (*p) {
        /* find opening quote of key */
        while (*p && *p != '"') ++p;
        if (!*p) break;
        const char* ks = ++p;
        while (*p && *p != '"') ++p;
        if (!*p) break;
        std::string key(ks, p - ks);
        ++p; /* skip closing quote */

        /* skip to ':' */
        while (*p && *p != ':') ++p;
        if (!*p) break;
        ++p;

        /* find opening quote of value */
        while (*p && *p != '"') ++p;
        if (!*p) break;
        const char* vs = ++p;
        while (*p && *p != '"') ++p;
        if (!*p) break;
        std::string val(vs, p - vs);
        ++p;

        if (!key.empty() && !val.empty())
            g_expert_map[key] = val;
    }
    fprintf(stderr, "[SODA/NVBit] Loaded %zu expert kernel entries\n",
            g_expert_map.size());
}

/* -----------------------------------------------------------------------
 * Receiver thread
 *
 * Drains the device→host channel and accumulates cache-line indices
 * per grid_launch_id.
 * --------------------------------------------------------------------- */
void* recv_thread_fun(void* args) {
    CUcontext ctx = (CUcontext)args;
    pthread_mutex_lock(&mutex);
    CTXstate* ctx_state = ctx_state_map[ctx];
    ChannelHost* ch_host = &ctx_state->channel_host;
    pthread_mutex_unlock(&mutex);

    char* buf = (char*)malloc(CHANNEL_SIZE);

    while (ctx_state->recv_thread_done == RecvThreadState::WORKING) {
        uint32_t nb = ch_host->recv(buf, CHANNEL_SIZE);
        if (nb > 0) {
            uint32_t off = 0;
            while (off < nb) {
                cl_access_t* acc = (cl_access_t*)(buf + off);
                pthread_mutex_lock(&mutex);
                auto& inv = g_invoc_map[acc->grid_launch_id];
                inv.cacheline_set.insert(acc->cacheline_index);
                inv.global_load_count++;
                pthread_mutex_unlock(&mutex);
                off += sizeof(cl_access_t);
            }
        }
    }
    free(buf);
    ctx_state->recv_thread_done = RecvThreadState::FINISHED;
    return nullptr;
}

/* -----------------------------------------------------------------------
 * Emit one JSON record for a completed invocation
 * --------------------------------------------------------------------- */
static void emit_invoc(uint64_t lid) {
    pthread_mutex_lock(&mutex);
    auto it = g_invoc_map.find(lid);
    if (it == g_invoc_map.end() || it->second.expert_type.empty()) {
        pthread_mutex_unlock(&mutex);
        return;
    }
    InvocData inv = std::move(it->second);
    g_invoc_map.erase(it);
    pthread_mutex_unlock(&mutex);

    if (!g_log_fp) return;

    fprintf(g_log_fp,
            "{\"kernel_name\": \"%s\", \"expert_type\": \"%s\", "
            "\"invocation\": %lu, \"global_load_count\": %lu, "
            "\"cacheline_set_size\": %zu, \"cacheline_hashes\": [",
            inv.kernel_name.c_str(), inv.expert_type.c_str(),
            (unsigned long)lid,
            (unsigned long)inv.global_load_count,
            inv.cacheline_set.size());

    size_t cnt = 0;
    for (uint64_t h : inv.cacheline_set) {
        if (cnt > 0) fprintf(g_log_fp, ", ");
        fprintf(g_log_fp, "%lu", (unsigned long)h);
        if (++cnt >= MAX_CL_HASHES_JSON) break;
    }
    fprintf(g_log_fp, "]}\n");
    fflush(g_log_fp);
}

/* -----------------------------------------------------------------------
 * Instrument a function: inject instrument_cacheline before LD.GLOBAL
 * --------------------------------------------------------------------- */
static void instrument_function_if_needed(CUcontext ctx, CUfunction func,
                                           ChannelDev* ch_dev) {
    /* Also instrument called device functions */
    std::vector<CUfunction> related = nvbit_get_related_functions(ctx, func);
    related.push_back(func);

    for (auto f : related) {
        if (!already_instrumented.insert(f).second) continue;

        const std::vector<Instr*>& instrs = nvbit_get_instrs(ctx, f);
        for (auto instr : instrs) {
            /* Only global load memory references */
            if (instr->getMemorySpace() != InstrType::MemorySpace::GLOBAL)
                continue;

            int mref_idx = 0;
            for (int i = 0; i < instr->getNumOperands(); i++) {
                const InstrType::operand_t* op = instr->getOperand(i);
                if (op->type != InstrType::OperandType::MREF) continue;

                nvbit_insert_call(instr, "instrument_cacheline", IPOINT_BEFORE);
                /* arg0: predicate */
                nvbit_add_call_arg_guard_pred_val(instr);
                /* arg1: memory address (64-bit) */
                nvbit_add_call_arg_mref_addr64(instr, mref_idx);
                /* arg2: grid_launch_id (set at launch time, slot 0) */
                nvbit_add_call_arg_launch_val64(instr, 0);
                /* arg3: channel device pointer */
                nvbit_add_call_arg_const_val64(instr, (uint64_t)ch_dev);
                mref_idx++;
            }
        }
    }
}

/* -----------------------------------------------------------------------
 * NVBit lifecycle callbacks
 * --------------------------------------------------------------------- */

void nvbit_at_init() {
    /* Set up mutexes as recursive */
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&mutex, &attr);
    pthread_mutex_init(&cuda_event_mutex, &attr);

    const char* map_path = getenv("SODA_EXPERT_MAP");
    if (!map_path) {
        fprintf(stderr,
                "[SODA/NVBit] Error: SODA_EXPERT_MAP env var not set\n");
        exit(1);
    }
    load_expert_map(map_path);

    const char* log_path = getenv("SODA_NVBIT_LOG");
    if (!log_path) {
        fprintf(stderr,
                "[SODA/NVBit] Error: SODA_NVBIT_LOG env var not set\n");
        exit(1);
    }
    g_log_fp = fopen(log_path, "w");
    if (!g_log_fp) {
        fprintf(stderr,
                "[SODA/NVBit] Error: cannot open log: %s\n", log_path);
        exit(1);
    }
}

void nvbit_at_ctx_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    CTXstate* s = new CTXstate;
    s->id = (int)ctx_state_map.size();
    ctx_state_map[ctx] = s;

    /* Load flush_channel fatbin (embedded by bin2c) */
    nvbit_load_tool_module(ctx, (const void*)flush_channel_bin,
                           &s->tool_module);
    nvbit_find_function_by_name(ctx, s->tool_module, "flush_channel",
                                &s->flush_channel_func);
    pthread_mutex_unlock(&mutex);
}

/* Called after nvbit_at_ctx_init — start receiver thread here */
void nvbit_tool_init(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    CTXstate* s = ctx_state_map[ctx];
    s->recv_thread_done = RecvThreadState::WORKING;
    cudaMallocManaged(&s->channel_dev, sizeof(ChannelDev));
    s->channel_host.init(s->id, CHANNEL_SIZE, s->channel_dev,
                         recv_thread_fun, ctx);
    nvbit_set_tool_pthread(s->channel_host.get_thread());
    pthread_mutex_unlock(&mutex);
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit,
                         nvbit_api_cuda_t cbid, const char* name,
                         void* params, CUresult* pStatus) {
    pthread_mutex_lock(&cuda_event_mutex);
    if (skip_callback_flag) {
        pthread_mutex_unlock(&cuda_event_mutex);
        return;
    }
    skip_callback_flag = true;

    CTXstate* s = ctx_state_map[ctx];

    /* Handle all kernel launch variants */
    bool is_launch = (cbid == API_CUDA_cuLaunchKernel      ||
                      cbid == API_CUDA_cuLaunchKernel_ptsz ||
                      cbid == API_CUDA_cuLaunchKernelEx     ||
                      cbid == API_CUDA_cuLaunchKernelEx_ptsz||
                      cbid == API_CUDA_cuLaunchCooperativeKernel ||
                      cbid == API_CUDA_cuLaunchCooperativeKernel_ptsz ||
                      cbid == API_CUDA_cuLaunchGridAsync    ||
                      cbid == API_CUDA_cuLaunch             ||
                      cbid == API_CUDA_cuLaunchGrid);

    if (is_launch) {
        CUfunction func;
        if (cbid == API_CUDA_cuLaunchKernelEx ||
            cbid == API_CUDA_cuLaunchKernelEx_ptsz) {
            func = ((cuLaunchKernelEx_params*)params)->f;
        } else if (cbid == API_CUDA_cuLaunch ||
                   cbid == API_CUDA_cuLaunchGrid) {
            func = ((cuLaunch_params*)params)->f;
        } else if (cbid == API_CUDA_cuLaunchGridAsync) {
            func = ((cuLaunchGridAsync_params*)params)->f;
        } else {
            func = ((cuLaunchKernel_params*)params)->f;
        }

        const char* kname = nvbit_get_func_name(ctx, func);
        auto et_it = g_expert_map.find(std::string(kname));
        bool is_expert = (et_it != g_expert_map.end());

        if (!is_exit) {
            if (is_expert) {
                /* Instrument this function (idempotent) */
                instrument_function_if_needed(ctx, func, s->channel_dev);

                /* Set launch-time argument (slot 0 = grid_launch_id) */
                nvbit_set_at_launch(ctx, func, (uint64_t)g_grid_launch_id);

                /* Record name/type for this launch ID */
                pthread_mutex_lock(&mutex);
                auto& inv = g_invoc_map[g_grid_launch_id];
                inv.kernel_name = std::string(kname);
                inv.expert_type = et_it->second;
                pthread_mutex_unlock(&mutex);

                /* Enable instrumented SASS */
                nvbit_enable_instrumented(ctx, func, true);
            } else {
                nvbit_enable_instrumented(ctx, func, false);
            }
            s->need_sync = true;

        } else { /* is_exit */
            if (is_expert) {
                /* Flush channel so recv thread sees all packets before emit */
                skip_callback_flag = false;
                pthread_mutex_unlock(&cuda_event_mutex);

                cudaDeviceSynchronize();
                void* args[] = {&s->channel_dev};
                nvbit_launch_kernel(ctx, s->flush_channel_func,
                                    1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
                cudaDeviceSynchronize();

                /* Spin until recv thread has drained all packets for this launch */
                pthread_mutex_lock(&mutex);
                bool pending = (g_invoc_map.count(g_grid_launch_id) &&
                                g_invoc_map[g_grid_launch_id].global_load_count > 0);
                pthread_mutex_unlock(&mutex);
                (void)pending;

                emit_invoc(g_grid_launch_id);
                g_grid_launch_id++;

                pthread_mutex_lock(&cuda_event_mutex);
                skip_callback_flag = true;
            } else {
                /* Non-expert: still advance launch counter */
                g_grid_launch_id++;
            }
        }
    }

    skip_callback_flag = false;
    pthread_mutex_unlock(&cuda_event_mutex);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    pthread_mutex_lock(&mutex);
    skip_callback_flag = true;
    CTXstate* s = ctx_state_map[ctx];

    if (s->need_sync) {
        void* args[] = {&s->channel_dev};
        nvbit_launch_kernel(ctx, s->flush_channel_func,
                            1, 1, 1, 1, 1, 1, 0, nullptr, args, nullptr);
        cudaDeviceSynchronize();
    }

    if (s->recv_thread_done != RecvThreadState::INIT) {
        s->recv_thread_done = RecvThreadState::STOP;
        while (s->recv_thread_done != RecvThreadState::FINISHED);
    }

    s->channel_host.destroy(false);
    cudaFree(s->channel_dev);
    delete s;
    ctx_state_map.erase(ctx);
    skip_callback_flag = false;
    pthread_mutex_unlock(&mutex);
}
