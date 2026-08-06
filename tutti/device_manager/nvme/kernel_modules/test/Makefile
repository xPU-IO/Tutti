# Build the standalone SNVMe sanity tests.
#
# Five tests plus one reset helper:
#   snvme_smoke           -- UAPI smoke, libc-only. Always built.
#   snvme_smoke_gpu       -- adds the NVM_MAP_DEVICE_*  GPU paths via CUDA.
#                            Built automatically if `nvcc` is on PATH.
#   snvme_smoke_qgroup    -- exercises NVM_CREATE_QUEUE_GROUP /
#                            NVM_DESTROY_QUEUE_GROUP container lifecycle.
#                            Always built; does NOT bind (safe to run on
#                            mounted devices).
#   snvme_smoke_addq      -- B3: exercises NVM_ADD_USER_QUEUE end-to-end
#                            (Create I/O CQ + Create I/O SQ + cascade
#                            destroy through NVM_DESTROY_QUEUE_GROUP).
#                            Always built; BINDS the controller (destructive).
#   snvme_smoke_io        -- B3 + IO: same as smoke_addq but additionally
#                            mmaps BAR0, hand-builds NVMe Read/Write SQEs,
#                            rings doorbells, and verifies the round-trip
#                            data integrity LBA-by-LBA on the live disk.
#                            Always built; BINDS AND WRITES TO LBA
#                            (destructive -- target disk must be empty).
#   snvme_ubind           -- owner-side reset helper. Always built.
#
# All binaries depend on nothing but the SNVMe UAPI header
#   ../../libnvm/include/ioctl.h
#
# Usage:
#   make
#   sudo ./snvme_smoke 0000:50:00.0                # UAPI-smoke (safe, no GPU)
#   sudo ./snvme_smoke --bind 0000:50:00.0         # full bring-up (destructive)
#   sudo ./snvme_smoke_gpu 0000:50:00.0            # UAPI + GPU map paths (safe)
#   sudo ./snvme_smoke_gpu --bind 0000:50:00.0     # + run probe with GPU queues (destructive)
#   sudo ./snvme_smoke_qgroup 0000:50:00.0         # queue-group lifecycle (safe, no bind)
#   sudo ./snvme_smoke_addq 0000:50:00.0           # B3 ADD_USER_QUEUE (destructive: binds)
#   sudo ./snvme_smoke_io 0000:50:00.0             # B3 + IO round-trip (destructive: writes LBA)
#
# Override knobs:
#   CC, CFLAGS, NVCC, NVCCFLAGS, CUDA_ARCH
#
# GPU arch selection (CUDA_ARCH):
#   - default: auto-detect via `nvidia-smi --query-gpu=compute_cap`
#     so the build follows the host GPU (e.g. H20 -> sm_90,
#     A100 -> sm_80, V100 -> sm_70).
#   - fallback: sm_80 (Ampere A100), still accepted by every
#     CUDA toolkit from 11.0 through 13.x.
#   - override: `make CUDA_ARCH=sm_90` to force one explicitly.
#   The previous baseline hard-coded sm_70, which CUDA 13 has
#   dropped -- nvcc reports "Unsupported gpu architecture 'sm_70'"
#   on any toolkit >= 13.0 (e.g. /usr/local/cuda-13.0/).

TARGET          := snvme_smoke
TARGET_GPU      := snvme_smoke_gpu
TARGET_QGROUP   := snvme_smoke_qgroup
TARGET_ADDQ     := snvme_smoke_addq
TARGET_IO       := snvme_smoke_io
TARGET_UBIND    := snvme_ubind

UAPI_INC  := ../../libnvm/include
TUTTI_INC := ../../../../include

CFLAGS    ?= -O2 -Wall -Wextra -Wno-unused-parameter -std=c11 \
             -I$(UAPI_INC) -I$(TUTTI_INC)
LDFLAGS   ?=

NVCC       ?= nvcc

# Probe the host's primary GPU compute capability and translate
# "X.Y" (e.g. "9.0") into "sm_XY" (e.g. "sm_90").  Empty if
# nvidia-smi is missing, returns no GPUs, or returns a malformed
# value -- in which case CUDA_ARCH falls through to the static
# default below.
DETECTED_CC := $(shell nvidia-smi --query-gpu=compute_cap \
                       --format=csv,noheader 2>/dev/null \
                   | awk '{ gsub(/[[:space:].]/, ""); \
                            if ($$0 ~ /^[0-9]+$$/) { print; exit } }')
ifneq ($(DETECTED_CC),)
CUDA_ARCH  ?= sm_$(DETECTED_CC)
else
# Static fallback if auto-detect fails.  sm_80 is the lowest arch
# that every CUDA toolkit from 11.0 through 13.x accepts AND that
# covers the GPU families we actually ship on (A100, H100/H20).
CUDA_ARCH  ?= sm_80
endif

NVCCFLAGS  ?= -O2 -std=c++17 -arch=$(CUDA_ARCH) \
              -DTUTTI_USE_CUDA=1 -I$(UAPI_INC) -I$(TUTTI_INC) \
              -Xcompiler=-Wall,-Wextra,-Wno-unused-parameter

SRC         := snvme_smoke.c
SRC_GPU     := snvme_smoke_gpu.cu
SRC_QGROUP  := snvme_smoke_qgroup.c
SRC_ADDQ    := snvme_smoke_addq.c
SRC_IO      := snvme_smoke_io.c
SRC_UBIND   := snvme_ubind.c

CPU_TARGETS := $(TARGET) $(TARGET_QGROUP) $(TARGET_ADDQ) $(TARGET_IO) \
               $(TARGET_UBIND)

# Build the GPU test only if nvcc is available.
HAVE_NVCC := $(shell command -v $(NVCC) >/dev/null 2>&1 && echo yes || echo no)

.PHONY: all cpu gpu clean run run-gpu run-qgroup run-addq run-io

ifeq ($(HAVE_NVCC),yes)
all: cpu gpu
else
all: cpu
	@echo "[i] nvcc not found; skipping $(TARGET_GPU)."
	@echo "    Install CUDA Toolkit (or set NVCC=...) and re-run make to enable it."
endif

cpu: $(CPU_TARGETS)

gpu: $(TARGET_GPU)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(TARGET_GPU): $(SRC_GPU)
	@echo "[+] nvcc: CUDA_ARCH=$(CUDA_ARCH) (override with 'make CUDA_ARCH=sm_XX')"
	$(NVCC) $(NVCCFLAGS) -o $@ $<

$(TARGET_QGROUP): $(SRC_QGROUP)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(TARGET_ADDQ): $(SRC_ADDQ)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(TARGET_IO): $(SRC_IO)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

$(TARGET_UBIND): $(SRC_UBIND)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET) $(TARGET_GPU) $(TARGET_QGROUP) $(TARGET_ADDQ) $(TARGET_IO) \
	      $(TARGET_UBIND)

# Convenience: PCI=0000:50:00.0 make run [BIND=1]
run: $(TARGET)
	@if [ -z "$(PCI)" ]; then \
	    echo "usage: PCI=DDDD:BB:DD.F make run [BIND=1]"; exit 1; \
	fi; \
	if [ "$(BIND)" = "1" ]; then \
	    sudo ./$(TARGET) --bind $(PCI); \
	else \
	    sudo ./$(TARGET) $(PCI); \
	fi

# Convenience: PCI=0000:50:00.0 make run-gpu [BIND=1]
run-gpu: $(TARGET_GPU)
	@if [ -z "$(PCI)" ]; then \
	    echo "usage: PCI=DDDD:BB:DD.F make run-gpu [BIND=1]"; exit 1; \
	fi; \
	if [ "$(BIND)" = "1" ]; then \
	    sudo ./$(TARGET_GPU) --bind $(PCI); \
	else \
	    sudo ./$(TARGET_GPU) $(PCI); \
	fi

# Convenience: PCI=0000:50:00.0 make run-qgroup
# (queue-group lifecycle smoke; does NOT bind, so no destructive side
# effects -- safe to run on a host where the device is mounted.)
run-qgroup: $(TARGET_QGROUP)
	@if [ -z "$(PCI)" ]; then \
	    echo "usage: PCI=DDDD:BB:DD.F make run-qgroup"; exit 1; \
	fi; \
	sudo ./$(TARGET_QGROUP) $(PCI)

# Convenience: PCI=0000:50:00.0 make run-addq
# (B3 NVM_ADD_USER_QUEUE smoke; BINDS the controller -- destructive.)
run-addq: $(TARGET_ADDQ)
	@if [ -z "$(PCI)" ]; then \
	    echo "usage: PCI=DDDD:BB:DD.F make run-addq"; exit 1; \
	fi; \
	sudo ./$(TARGET_ADDQ) $(PCI)

# Convenience: PCI=0000:50:00.0 make run-io
# (B3 + IO round-trip smoke; BINDS the controller AND WRITES TO LBA --
#  destructive, target disk must be empty.)
run-io: $(TARGET_IO)
	@if [ -z "$(PCI)" ]; then \
	    echo "usage: PCI=DDDD:BB:DD.F make run-io"; exit 1; \
	fi; \
	sudo ./$(TARGET_IO) $(PCI)
