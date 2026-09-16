"""tutti_runtime — Python bindings for the Tutti StorageRuntime.

Public surface:
    make_stub_runtime(accel_id=-1) -> Runtime
    make_local_nvme_runtime(preset: dict) -> Runtime
    make_striped_nvme_runtime(preset: dict) -> Runtime
    Runtime: caps / open_batch / close_target / close_batch /
             register_memory / submit / release_io /
             wait / wait_result / wait_detail / shutdown / testing_force_complete
    SubmitResult: status_ok / status_msg / io_handle / initial_states / rejected
    WaitResult: structured terminal result retained after release_io
"""

from ._core import (
    Runtime,
    SubmitResult,
    WaitResult,
    make_local_nvme_runtime,
    make_stub_runtime,
    make_striped_nvme_runtime,
)

__all__ = [
    "Runtime",
    "SubmitResult",
    "WaitResult",
    "make_stub_runtime",
    "make_local_nvme_runtime",
    "make_striped_nvme_runtime",
]
