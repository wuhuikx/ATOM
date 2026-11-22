# SPDX-License-Identifier: MIT
# AsyncLLMEngine.weight_sync_session runs its body inside the core manager's
# session (GPU-free).

from contextlib import contextmanager

import pytest

from atom.rollout.async_engine import AsyncLLMEngine


class _CoreManager:
    def __init__(self):
        self.events = []

    @contextmanager
    def weight_sync_session(self):
        self.events.append("enter")
        try:
            yield
        finally:
            self.events.append("exit")


def _engine():
    engine = AsyncLLMEngine.__new__(AsyncLLMEngine)
    engine.core_mgr = _CoreManager()
    return engine


def test_weight_sync_session_wraps_body_in_core_manager_session():
    engine = _engine()
    with engine.weight_sync_session():
        engine.core_mgr.events.append("body")
    assert engine.core_mgr.events == ["enter", "body", "exit"]


def test_weight_sync_session_closes_core_manager_session_on_error():
    engine = _engine()
    with pytest.raises(RuntimeError, match="sender failed"):
        with engine.weight_sync_session():
            raise RuntimeError("sender failed")
    assert engine.core_mgr.events == ["enter", "exit"]
