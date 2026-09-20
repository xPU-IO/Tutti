"""跨层共享的异常类型。

放在 ``common/`` 而非 ``engine/``：``storage/`` 层需要抛出并捕获这些
异常，而 storage 是 engine 的下层——分层图要求底层不反向 import 上层。
"""

from __future__ import annotations


class DirectTransferUnavailable(RuntimeError):
    """Raised when a store cannot expose a direct paged-memory backend."""
