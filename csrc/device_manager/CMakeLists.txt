# tutti/device_manager/CMakeLists.txt
#
# Layer 2 container — shrunk by T-041 (Round 10 Session 2).
#
# The control-plane driver/common/mock sources that used to live here were
# deleted in 2026-08 dead-code cleanup (the production DataPath talks to
# libnvm directly; the driver abstraction layer had no consumer). What
# remains in this directory is the single-source-of-truth for libnvm, the
# gRPC nvmeservice daemon, and the snvme kernel_modules tree.
#
# Per Session 1: libnvm, nvmeservice, and kernel_modules stay at this location;
# the default hardware build includes all three.

# =============================================================================
# libnvm — user-space NVMe queue library (wraps snvme kernel module)
# =============================================================================
add_subdirectory(nvme/libnvm)

# =============================================================================
# nvmeservice — cross-process arbitration daemon
# =============================================================================
add_subdirectory(nvme/nvmeservice)

# kernel_modules/ contains the snvme kernel module sources; built out-of-tree
# by the deployment system (see kernel_modules/PORTING.md), not by CMake.
