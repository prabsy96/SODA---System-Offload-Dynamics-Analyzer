#include "utils/channel.hpp"

/* Flushed to device via nvbit_launch_kernel at context teardown */
extern "C" __global__ void flush_channel(ChannelDev* ch_dev) {
    ch_dev->flush();
}
