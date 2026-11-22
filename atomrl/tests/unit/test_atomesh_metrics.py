"""The trainer's per-step ATOMesh metrics, drained from the samples that
``AtomMeshRolloutClient`` records against a stub ATOM-behind-ATOMesh
``/generate`` that keeps a prefix cache, correctly or not across a sync."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from aiohttp import web
from atomrl.core.config import AtomRLConfig
from atomrl.engine.inference import atomesh_rollout
from atomrl.engine.inference.atomesh_rollout import AtomMeshRolloutClient
from atomrl.trainer.rl_trainer import RLTrainer
from tests.unit.test_atomesh_transport_config import (  # noqa: F401
    DIRECT,
    _atomesh,
    _data_root,
    _setup,
    ray_free,
)

SAMPLING = {"max_tokens": 8, "temperature": 1.0, "top_p": 1.0, "top_k": -1}
PROMPTS = [[11, 12, 13, 14], [21, 22]]
METRICS = {
    "rollout/atomesh_cached_tokens_mean",
    "rollout/atomesh_prefix_hit_rate",
    "rollout/atomesh_post_sync_cached_tokens",
    "rollout/atomesh_weight_version_lag_max",
}


class CachingEngine:
    """A stub ATOMesh in front of one worker with a prefix cache.

    A request whose prompt is cached reports all but its last prompt token
    as cached; one that is not caches it. A weight sync (``sync``) drops the
    cache, unless ``keeps_old_prefix``. The request that fills the cache
    answers last, so completion order never puts it first.
    """

    def __init__(self, keeps_old_prefix: bool = False) -> None:
        self.keeps_old_prefix = keeps_old_prefix
        self.version = 0
        self.cached: set[tuple[int, ...]] = set()
        self._loop = asyncio.new_event_loop()
        self._runner: web.AppRunner | None = None
        self.url = ""

    def sync(self, version: int) -> None:
        self.version = version
        if not self.keeps_old_prefix:
            self.cached.clear()

    async def _generate(self, request: web.Request) -> web.Response:
        body = await request.json()
        prompt = tuple(body["input_ids"])
        hit = prompt in self.cached
        self.cached.add(prompt)
        if not hit:
            await asyncio.sleep(0.05)
        meta = {
            "id": f"req-{len(prompt)}",
            "finish_reason": {"type": "stop"},
            "cached_tokens": len(prompt) - 1 if hit else 0,
            "weight_version": str(self.version),
            "output_token_logprobs": [[-0.5, 7, None]],
        }
        return web.json_response({"text": "", "output_ids": [7], "meta_info": meta})

    async def _start(self) -> None:
        app = web.Application()
        app.router.add_post("/generate", self._generate)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"

    def __enter__(self) -> CachingEngine:
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        asyncio.run_coroutine_threadsafe(self._start(), self._loop).result()
        return self

    def __exit__(self, *exc: Any) -> None:
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)


def _rollout(client: AtomMeshRolloutClient, n: int = 4) -> None:
    prompts = [p for p in PROMPTS for _ in range(n)]
    asyncio.run(client.generate_batch(prompts, SAMPLING))


def _sync(engine: CachingEngine, client: AtomMeshRolloutClient, version: int):
    engine.sync(version)
    client.committed_version = version


@pytest.mark.parametrize("keeps_old_prefix", [False, True])
def test_post_sync_cached_tokens_catches_an_old_prefix_hit(keeps_old_prefix):
    with CachingEngine(keeps_old_prefix) as engine:
        client = AtomMeshRolloutClient(engine.url)
        _rollout(client)
        before = client.drain_sample_meta()
        _sync(engine, client, 1)
        _rollout(client)
        after = client.drain_sample_meta()

    # Before any sync no prompt has an earlier version to compare with.
    assert before["post_sync_cached_tokens"] == 0.0
    if keeps_old_prefix:
        # Every sample of both prompts hits v0's prefix: 3 + 1 tokens.
        assert after["post_sync_cached_tokens"] == 4.0
    else:
        assert after["post_sync_cached_tokens"] == 0.0


def test_a_prompt_seen_again_without_a_sync_is_not_post_sync():
    with CachingEngine(keeps_old_prefix=True) as engine:
        client = AtomMeshRolloutClient(engine.url)
        _rollout(client)
        client.drain_sample_meta()
        _rollout(client)
        stats = client.drain_sample_meta()
    assert stats["post_sync_cached_tokens"] == 0.0
    assert stats["prefix_hit_rate"] == pytest.approx((4 * 3 + 4 * 1) / (4 * 4 + 4 * 2))


def test_drain_empties_sample_meta_and_summarises_it():
    with CachingEngine() as engine:
        client = AtomMeshRolloutClient(engine.url)
        _rollout(client)
        assert len(client.sample_meta) == 8
        client.committed_version = 2
        stats = client.drain_sample_meta()
        assert client.sample_meta == []
        assert client.drain_sample_meta() == {}

    # Per prompt, the first sample filled the cache and three hit it.
    assert stats["cached_tokens_mean"] == pytest.approx((3 * 3 + 3 * 1) / 8)
    assert stats["prefix_hit_rate"] == pytest.approx((3 * 3 + 3 * 1) / 24)
    assert stats["weight_version_lag_max"] == 2.0


def test_each_step_logs_the_four_metrics_and_drains(
    ray_free, monkeypatch  # noqa: F811
):
    with CachingEngine(keeps_old_prefix=True) as engine:

        class Client(AtomMeshRolloutClient):
            def __init__(self, base_url: str, **kwargs: Any) -> None:
                super().__init__(engine.url, **kwargs)

            async def join_weight_group(self, trainer, **kwargs: Any) -> dict:
                return {"world_size": 5}

        monkeypatch.setattr(atomesh_rollout, "AtomMeshRolloutClient", Client)
        trainer = _setup(_atomesh())
        client = trainer._atomesh_client
        trainer._rollout_with_ray_vllm(["ab", "c"], 2)
        first = trainer._atomesh_rollout_metrics()
        _sync(engine, client, 1)
        trainer._rollout_with_ray_vllm(["ab", "c"], 2)
        second = trainer._atomesh_rollout_metrics()

    assert set(first) == set(second) == METRICS
    assert client.sample_meta == []
    assert first["rollout/atomesh_post_sync_cached_tokens"] == 0.0
    assert second["rollout/atomesh_post_sync_cached_tokens"] > 0.0


def test_no_other_transport_logs_them(ray_free):  # noqa: F811
    trainer = RLTrainer(AtomRLConfig.from_yaml(DIRECT))
    assert trainer._atomesh_client is None
    assert trainer._atomesh_rollout_metrics() == {}
