"""GPU-direct page IO for independently allocated, block-addressable KV entities."""
from dataclasses import dataclass
import logging

from .store import DirectAdmissionError, DirectPoolGeometry, TuttiDirectBackend

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerPageGeometry:
    page_bytes: int
    block_stride: int
    num_blocks: int


def page_geometry(tensor, page_bytes: int) -> LayerPageGeometry:
    """MiniMax NVIDIA packed NHD pages and contiguous key-only index pages."""
    shape = tuple(map(int, tensor.shape))
    strides = tuple(int(s) * tensor.element_size() for s in tensor.stride())
    size = tensor.numel() * tensor.element_size()
    if page_bytes <= 0 or page_bytes % 4096 or size % page_bytes:
        raise DirectAdmissionError("layer page and storage must be 4KiB aligned")
    blocks = size // page_bytes
    if len(shape) not in (3, 4) or blocks <= 0 or shape[0] <= 0:
        raise DirectAdmissionError(f"packed block-axis-0 KV layout required: {shape}")
    element = tensor.element_size()
    if len(shape) == 3:
        expected = (shape[1] * shape[2] * element, shape[2] * element, element)
    else:
        # Logical [blocks, heads, tokens, packed_content], physical NHD.
        expected = (shape[1] * shape[2] * shape[3] * element,
                    shape[3] * element, shape[1] * shape[3] * element, element)
    if any(count > 1 and stride != required
           for count, stride, required in zip(shape, strides, expected)):
        raise DirectAdmissionError("layer page must have compact packed NHD bytes")
    kernel_page_bytes = expected[0]
    if (kernel_page_bytes <= 0 or strides[0] != kernel_page_bytes or
            page_bytes % kernel_page_bytes or shape[0] % (page_bytes // kernel_page_bytes)):
        raise DirectAdmissionError("layer block stride does not form complete logical pages")
    return LayerPageGeometry(page_bytes, page_bytes, blocks)


class TuttiLayerwiseDirectBackend(TuttiDirectBackend):
    """Per-entity page geometry; fixed disk segments may contain unused padding."""

    # The partial-submit state machine is coupled to layerwise callback
    # progression. Detached request feeders use the stable load_layer API and
    # poll its completion handle instead, preserving request-level overlap
    # without parking a request on a foreground-only retry state.
    supports_detached_try_read = False

    def __init__(self, store):
        super().__init__(store)
        self._layer_pages = ()
        self._layer_tickets = []
        self._layer_tensors = ()

    def register_paged_caches(self, pool, *, num_layers, blocks_per_chunk,
                              chunk_tokens, segment_bytes, max_chunks_per_wave=None):
        if self._layer_tickets or self._memory_ticket is not None:
            raise DirectAdmissionError("layerwise cache already registered")
        if not isinstance(pool, dict) or len(pool) != num_layers:
            raise DirectAdmissionError("layerwise cache entity count mismatch")
        page_sizes = self._store._layer_page_bytes
        if len(page_sizes) != num_layers or max(page_sizes) * blocks_per_chunk != segment_bytes:
            raise DirectAdmissionError("layerwise physical pages do not match disk geometry")
        tensors = tuple(pool.values())
        self._layer_tensors = tensors
        pages = tuple(page_geometry(t, p) for t, p in zip(tensors, page_sizes))
        device = tensors[0].get_device()
        if any(not t.is_cuda or t.get_device() != device for t in tensors):
            raise DirectAdmissionError("all layerwise caches must reside on the bound CUDA device")
        if len({p.num_blocks for p in pages}) != 1:
            raise DirectAdmissionError("layerwise caches must share logical block IDs")
        geometry = DirectPoolGeometry(0, sum(t.numel() * t.element_size() for t in tensors),
                                      0, 0, max(page_sizes), pages[0].num_blocks, num_layers,
                                      chunk_tokens // blocks_per_chunk, blocks_per_chunk,
                                      segment_bytes, device)
        self._check_runtime_capacity(geometry, max_chunks_per_wave=max_chunks_per_wave)
        try:
            for tensor in tensors:
                size = tensor.numel() * tensor.element_size()
                if tensor.data_ptr() % 65536:
                    raise DirectAdmissionError("layerwise cache base must be 64KiB aligned; no extra KV copy is allocated")
                ticket = self._store._runtime.register_memory(
                    tensor.data_ptr(), size, "device", accel_id=device, io_granularity=0)
                self._layer_tickets.append(int(ticket))
        except Exception:
            try:
                self.close()
            except Exception:
                _LOG.exception("Layerwise registration cleanup failed; ownership retained")
            raise
        self._pool = tensors
        self._geometry = geometry
        self._layer_pages = pages
        self._memory_ticket = self._layer_tickets[0]
        _LOG.warning("DIRECT_LAYERWISE_ACCEPTED entities=%d blocks=%d physical_pages=%s disk_segment=%d accel=%d",
                     num_layers, geometry.num_blocks, page_sizes, segment_bytes, device)
        return True

    def _page_requests(self, layer_idx, block_id, block_ordinal, entry, plan):
        page = self._layer_pages[layer_idx]
        base = entry.target_offset + layer_idx * self.geometry.segment_bytes + block_ordinal * page.page_bytes
        return [(entry.target_ticket, base, self._layer_tickets[layer_idx],
                 block_id * page.block_stride, page.page_bytes, plan.direction)]

    def warm_up_registration(self):
        # Each entity has its own memory ticket/domain registration.
        geometry = self._geometry
        if geometry is None:
            return False
        ticket = next((self._store._targets[u].ticket for u in self._store._warm_uris
                       if u in self._store._targets and self._store._targets[u].ticket), 0)
        if not ticket:
            return False
        from .store import _TuttiCompletion

        for memory in self._layer_tickets:
            completion = _TuttiCompletion(self._store._runtime,
                self._store._submit_retry([(ticket, 0, memory, 0, 4096, "read")], "read"),
                lambda _ok: None, auto_watch=False, direction="read")
            completion.wait_result()
        return True

    def close(self):
        # Parent drains all plans and closes targets before unregistering the
        # first ticket. Keep every remaining backing alive if release fails.
        if self._closed:
            return
        if self._memory_ticket is None and self._layer_tickets:
            self._memory_ticket = self._layer_tickets[0]
        first = self._memory_ticket
        first_error = None
        try:
            super().close()
        except Exception as exc:
            if self._memory_ticket is not None:
                raise
            first_error = exc
        if first in self._layer_tickets:
            self._layer_tickets.remove(first)
        self._closed = False
        while self._layer_tickets:
            try:
                self._store._runtime.unregister_memory(self._layer_tickets[-1])
            except Exception:
                self._memory_ticket = self._layer_tickets[0]
                raise
            self._layer_tickets.pop()
        self._layer_tensors = ()
        self._closed = True
        if first_error is not None:
            raise first_error
