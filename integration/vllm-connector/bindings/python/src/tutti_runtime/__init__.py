"""tutti_runtime — Python bindings for the Tutti StorageRuntime.

Public surface:
    make_stub_runtime(accel_id=-1) -> Runtime
    make_local_nvme_runtime(preset: dict) -> Runtime
    make_striped_nvme_runtime(preset: dict) -> Runtime
    Runtime: caps / open_batch / register_memory / submit / release_io /
             wait / shutdown / testing_force_complete
    SubmitResult: status_ok / status_msg / io_handle / initial_states / rejected
"""

from ._core import (
    Runtime,
    SubmitResult,
    make_local_nvme_runtime,
    make_stub_runtime,
    make_striped_nvme_runtime,
)

__all__ = [
    "Runtime",
    "SubmitResult",
    "make_stub_runtime",
    "make_local_nvme_runtime",
    "make_striped_nvme_runtime",
]
