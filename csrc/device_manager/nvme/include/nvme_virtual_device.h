#pragma once
// Redirect shim — the implementation has been relocated to the
// LocalNvmeDataPath control package (T-041 / Round 10 Session 2).
//
// This header exists ONLY so that legacy Layer-3 backend sources which use the
// path-qualified form  #include "device_manager/nvme/include/nvme_virtual_device.h"
// (resolved via the project-source include root) keep compiling without a
// source edit. New code must include the real header directly:
//   #include "data_paths/local_nvme/control/nvme/include/nvme_virtual_device.h"
//   or, via the control target's include dir, the bare form:
//   #include "nvme_virtual_device.h"
#include "data_paths/local_nvme/control/nvme/include/nvme_virtual_device.h"
