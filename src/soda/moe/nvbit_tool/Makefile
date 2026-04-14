# Makefile for mem_reuse_tracker NVBit tool
#
# Usage:
#   make NVBIT_PATH=/path/to/nvbit_release_x86_64/core
#   # or: export NVBIT_PATH before calling make
#
# Output: mem_reuse_tracker.so
#
# Requirements:
#   - nvcc (CUDA 10.1+)
#   - bin2c  (ships with CUDA toolkit, same bin dir as nvcc)
#   - ptxas  (ships with CUDA toolkit)
#   - NVBIT_PATH set to the 'core' dir inside the NVBit release

ifndef NVBIT_PATH
  ifdef NVBIT_SDK_PATH
    # Accept NVBIT_SDK_PATH (the tarball root) as an alias
    NVBIT_PATH := $(NVBIT_SDK_PATH)/core
  else
    # Default: use the bundled SDK in this repo
    NVBIT_PATH := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))nvbit_release_x86_64/core
  endif
endif

NVCC    := nvcc -ccbin=$(CXX) -D_FORCE_INLINES
PTXAS   := ptxas
BIN2C   := bin2c

NVCC_OPT := -O3

# Detect ptxas version; older ptxas needs -maxrregcount=24 for inject_funcs
PTXAS_VER_ADD_FLAG := 12.3
PTXAS_VER          := $(shell $(PTXAS) --version 2>/dev/null | grep release | cut -f2 -d, | cut -f3 -d' ')
PTXAS_VER_CHECK    := $(shell echo "${PTXAS_VER} >= $(PTXAS_VER_ADD_FLAG)" | bc 2>/dev/null)
ifeq ($(PTXAS_VER_CHECK), 0)
  MAXRREGCOUNT_FLAG := -maxrregcount=24
else
  MAXRREGCOUNT_FLAG :=
endif

INCLUDES := -I$(NVBIT_PATH)
LIBS     := -L$(NVBIT_PATH) -lnvbit
CUDA_LIB := -L$(subst bin/nvcc,lib64,$(shell which nvcc | tr -s /))
ARCH     ?= all

TOOL     := mem_reuse_tracker.so

all: $(TOOL)

# -----------------------------------------------------------------------
# Step 1: compile flush_channel.cu → .fatbin → .c (embedded binary)
# -----------------------------------------------------------------------
tool_func/flush_channel.fatbin: tool_func/flush_channel.cu
	$(NVCC) -arch=$(ARCH) $(INCLUDES) $(NVCC_OPT) -fatbin $< -o $@

tool_func/flush_channel.c: tool_func/flush_channel.fatbin
	$(BIN2C) -c --name flush_channel_bin $< > $@

# -----------------------------------------------------------------------
# Step 2: compile inject_funcs.cu (device-side, special flags)
# -----------------------------------------------------------------------
inject_funcs.o: inject_funcs.cu
	$(NVCC) $(INCLUDES) $(MAXRREGCOUNT_FLAG) \
	    -Xptxas -astoolspatch --keep-device-functions \
	    -arch=$(ARCH) -Xcompiler -Wall -Xcompiler -fPIC \
	    -c $< -o $@

# -----------------------------------------------------------------------
# Step 3: compile mem_reuse_tracker.cu (host-side tool, depends on .c embed)
# -----------------------------------------------------------------------
mem_reuse_tracker.o: mem_reuse_tracker.cu tool_func/flush_channel.c
	$(NVCC) -dc -c -std=c++11 $(INCLUDES) \
	    -Xptxas -cloning=no -Xcompiler -Wall \
	    -arch=$(ARCH) $(NVCC_OPT) -Xcompiler -fPIC \
	    $< -o $@

# -----------------------------------------------------------------------
# Step 4: link
# -----------------------------------------------------------------------
$(TOOL): mem_reuse_tracker.o inject_funcs.o $(NVBIT_PATH)/libnvbit.a
	$(NVCC) -arch=$(ARCH) $(NVCC_OPT) \
	    mem_reuse_tracker.o inject_funcs.o \
	    $(LIBS) $(CUDA_LIB) -lcuda -lcudart_static \
	    -shared -o $@
	@echo ""
	@echo "Built: $(TOOL)"
	@echo "Usage: LD_PRELOAD=./$(TOOL) python <inference_script.py>"

clean:
	rm -f *.so *.o tool_func/*.fatbin tool_func/*.c

.PHONY: all clean
