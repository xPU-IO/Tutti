# SED rewrite rules to convert upstream nvme-5.4.241 host driver symbols
# into the snvme-coexistence flavour for kernel 5.4.
#
# Scope of this rename pass (minimal):
#   Rename ONLY the global identifiers that snvme exports from snvme-core.ko
#   AND that the in-tree nvme-core.ko also exports under the same name -- if
#   we did not rename them, loading both modules concurrently would fail
#   with "exports duplicate symbol" or the kernel module loader would refuse
#   to bind the second copy.  Function-internal helpers and non-exported
#   helpers are intentionally left under their upstream names.
#
# How this list was derived:
#   - Cross-referenced `EXPORT_SYMBOL_GPL(...)` from
#     temp/nvme-5.4.241/host/{core,fabrics,multipath,rdma,tcp}.c
#   - Kept identifiers that snvme-5.15.0 also renames (so the two baselines
#     present a consistent symbol surface to the rest of snvme).
#   - Added `nvme_init_identify` because it is exported in 5.4 but was
#     replaced upstream by `nvme_init_ctrl_finish` in 5.15 -- 5.4-specific
#     conflict that snvme-5.15.0 did not need to solve.
#   - Removed entries that exist only in 5.15 and would be no-ops here:
#     `nvme_alloc_request_qid`, `nvme_init_ctrl_finish`,
#     `__nvme_check_ready` / `nvme_check_ready`,
#     `nvme_fail_nonready_command`.
#
# Identifiers intentionally NOT renamed (must keep upstream name):
#   - `nvme_ctrl`, `nvme_ns`, `nvme_command`, `nvme_request`, `nvme_req`,
#     `nvme_dev`, `nvme_queue`, `nvme_ctrl_state` and other types -- these
#     are only structurally visible inside the module and never exported.
#   - `nvme_reset_ctrl_sync`, `nvme_delete_ctrl`, `nvme_cancel_tagset`,
#     `nvme_cancel_admin_tagset`, `nvme_stop_keep_alive`,
#     `nvme_sync_io_queues` -- snvme-5.15.0 keeps the upstream name and
#     simply removes their EXPORT_SYMBOL_GPL.  We do the same here for
#     consistency.
#   - `admin_timeout` -- fully renamed here (definition, EXPORT, the
#     nvme.h extern, the ADMIN_TIMEOUT macro body, and the single
#     open-coded `admin_timeout * 1000` reference in the firmware
#     activation timeout).  This diverges from snvme-5.15.0, which
#     renames only the definition/EXPORT and leaves the nvme.h extern
#     + macro body referencing the old name -- a latent unresolved
#     symbol that snvme-5.15.0 apparently resolves at module load time
#     against the in-tree nvme-core.ko's exported `admin_timeout`.
#     We do NOT want that implicit cross-module dependency; treating
#     `admin_timeout` as a single bare-identifier global that gets
#     uniformly renamed is safe here because there is no syntactic
#     ambiguity with any other identifier in the tree.

# -- module-private global workqueues + module params --
s/\bnvme_wq\b/s_nvme_wq/g
s/\bnvme_reset_wq\b/s_nvme_reset_wq/g
s/\bnvme_delete_wq\b/s_nvme_delete_wq/g
s/\bnvme_io_timeout\b/s_nvme_io_timeout/g
s/\bnvme_max_retries\b/s_nvme_max_retries/g
s/\badmin_timeout\b/s_admin_timeout/g

# -- functions exported from snvme-core.ko, kept alphabetical --
s/\bnvme_alloc_request\b/snvme_alloc_request/g
s/\bnvme_cancel_request\b/snvme_cancel_request/g
s/\bnvme_change_ctrl_state\b/snvme_change_ctrl_state/g
s/\bnvme_cleanup_cmd\b/snvme_cleanup_cmd/g
s/\bnvme_complete_async_event\b/snvme_complete_async_event/g
s/\bnvme_complete_rq\b/snvme_complete_rq/g
s/\bnvme_disable_ctrl\b/snvme_disable_ctrl/g
s/\bnvme_enable_ctrl\b/snvme_enable_ctrl/g
s/\bnvme_find_get_ns\b/snvme_find_get_ns/g
s/\bnvme_get_features\b/snvme_get_features/g
# 5.4-specific: the identify path is still a separate exported symbol in
# 5.4; in 5.15 it was folded into nvme_init_ctrl_finish.
s/\bnvme_init_identify\b/snvme_init_identify/g
s/\bnvme_init_ctrl\b/snvme_init_ctrl/g
s/\bnvme_kill_queues\b/snvme_kill_queues/g
s/\bnvme_put_ns\b/snvme_put_ns/g
s/\bnvme_remove_namespaces\b/snvme_remove_namespaces/g
s/\bnvme_reset_ctrl\b/snvme_reset_ctrl/g
s/\bnvme_sec_submit\b/snvme_sec_submit/g
s/\bnvme_set_features\b/snvme_set_features/g
s/\bnvme_set_queue_count\b/snvme_set_queue_count/g
s/\bnvme_setup_cmd\b/snvme_setup_cmd/g
s/\bnvme_shutdown_ctrl\b/snvme_shutdown_ctrl/g
s/\bnvme_start_ctrl\b/snvme_start_ctrl/g
s/\bnvme_start_freeze\b/snvme_start_freeze/g
s/\bnvme_start_queues\b/snvme_start_queues/g
s/\bnvme_stop_ctrl\b/snvme_stop_ctrl/g
s/\bnvme_stop_queues\b/snvme_stop_queues/g
s/\bnvme_submit_sync_cmd\b/snvme_submit_sync_cmd/g
s/\b__nvme_submit_sync_cmd\b/__snvme_submit_sync_cmd/g
s/\bnvme_sync_queues\b/snvme_sync_queues/g
s/\bnvme_try_sched_reset\b/snvme_try_sched_reset/g
s/\bnvme_unfreeze\b/snvme_unfreeze/g
s/\bnvme_uninit_ctrl\b/snvme_uninit_ctrl/g
s/\bnvme_wait_freeze_timeout\b/snvme_wait_freeze_timeout/g
s/\bnvme_wait_freeze\b/snvme_wait_freeze/g
s/\bnvme_wait_reset\b/snvme_wait_reset/g

# Note: the PCI driver-struct rename and char-device entry-point names
# (nvme_driver -> snvme_driver, snvme_helpers_cdev*) are intentionally
# NOT in this sed file.  Those identifiers only appear inside the snvme
# increment that has to be re-implanted into pci.c; that increment is
# already authored using the snvme_* names directly and does not need a
# rename pass.
