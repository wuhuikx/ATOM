"""Fully async rollout with partial rollout under ``atom_cfg.transport: atomesh``:
what the config accepts, the two async 4+4 configs, and a ``StreamingRollout``
over the trainer's ATOMesh engine that syncs weights while prompt groups are in
flight, against a stub ATOMesh served from its own thread."""

from __future__ import annotations

import asyncio
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any

import pytest
import torch
from aiohttp import web
from atomrl.core.config import AtomRLConfig
from atomrl.engine.inference.atomesh_rollout import WeightVersionMismatch
from atomrl.trainer import rl_trainer
from atomrl.trainer.rl_trainer import RLTrainer
from omegaconf import OmegaConf

CONFIGS = Path(__file__).resolve().parents[2] / "examples" / "DAPO" / "configs"
KEY = "rl-secret"
FULLY_ASYNC = "controller.ray.fully_async"


@pytest.fixture(autouse=True)
def _data_root(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/data")


def _atomesh(*overrides: str, url: str = "http://127.0.0.1:18400") -> AtomRLConfig:
    return AtomRLConfig.from_yaml(
        CONFIGS / "dapo_qwen3_8b_atom_bf16_4p4_atomesh.yaml",
        overrides=[f"policy.generation.atom_cfg.base_url={url}", *overrides],
    )


# ── config validation ──────────────────────────────────────────────────────


@pytest.mark.parametrize("partial", [True, False])
def test_atomesh_accepts_fully_async(partial):
    cfg = _atomesh(
        f"{FULLY_ASYNC}.enabled=true", f"{FULLY_ASYNC}.partial_rollout={partial}"
    )
    assert cfg.controller.ray.fully_async.enabled
    assert cfg.controller.ray.fully_async.partial_rollout is partial


def test_atomesh_still_refuses_one_step_off_policy():
    with pytest.raises(ValueError, match="one_step_off_policy must be false"):
        _atomesh("controller.ray.one_step_off_policy=true")


def _flat(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}

    def walk(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{prefix}{key}.")
        else:
            out[prefix[:-1]] = node

    walk(OmegaConf.to_container(OmegaConf.load(path), resolve=False), "")
    return out


@pytest.mark.parametrize("kind", ["direct", "atomesh"])
def test_the_async_configs_differ_from_the_sync_ones_in_fully_async_only(kind):
    sync = _flat(CONFIGS / f"dapo_qwen3_8b_atom_bf16_4p4_{kind}.yaml")
    path = CONFIGS / f"dapo_qwen3_8b_atom_bf16_4p4_{kind}_async.yaml"
    derived = _flat(path)
    changed = {k for k in sync.keys() | derived.keys() if sync.get(k) != derived.get(k)}
    # Besides the fully_async block, the run's own name and checkpoint directory.
    assert changed - {"checkpointing.checkpoint_dir", "logger.wandb.name"} == {
        f"{FULLY_ASYNC}.enabled",
        f"{FULLY_ASYNC}.trigger_parameter_sync_step",
        f"{FULLY_ASYNC}.staleness_threshold",
        f"{FULLY_ASYNC}.partial_rollout",
    }
    assert derived[f"{FULLY_ASYNC}.staleness_threshold"] == 1.0
    assert derived[f"{FULLY_ASYNC}.partial_rollout"] is True
    assert (
        derived["checkpointing.checkpoint_dir"] != sync["checkpointing.checkpoint_dir"]
    )
    overrides = ["policy.generation.vllm_cfg.enable_sleep_mode=false"]
    if kind == "atomesh":
        overrides.append("policy.generation.atom_cfg.base_url=http://mesh")
    assert AtomRLConfig.from_yaml(path, overrides).controller.ray.fully_async.enabled


# ── a stub ATOMesh on its own thread ───────────────────────────────────────


class StubMesh:
    """ATOMesh's ``/v1/rl`` routes and ``/generate`` over four TP1 workers.

    ``/generate`` holds every request until ``release()``, as a long rollout
    would, then answers with the weight version of the worker the prompt hashes
    to. A pause keeps the held requests; nothing is answered while paused.
    """

    def __init__(self, workers: int = 4):
        self.versions = {f"w{i}": "0" for i in range(workers)}
        self.log: list[tuple] = []
        self.in_flight = 0
        self.paused = False
        self._loop = asyncio.new_event_loop()
        self._release: asyncio.Event | None = None
        ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, args=(ready,), daemon=True)
        self._thread.start()
        assert ready.wait(10)

    def _serve(self, ready: threading.Event) -> None:
        asyncio.set_event_loop(self._loop)
        self._release = asyncio.Event()
        app = web.Application()
        app.router.add_get("/v1/rl/workers", self._workers)
        app.router.add_post("/v1/rl/workers/{id}/engine/{path}", self._join)
        app.router.add_route("*", "/v1/rl/engine/{path}", self._fanout)
        app.router.add_post("/generate", self._generate)
        self._runner = web.AppRunner(app)
        self._loop.run_until_complete(self._runner.setup())
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        self._loop.run_until_complete(site.start())
        self.url = f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"
        ready.set()
        self._loop.run_forever()

    def release(self) -> None:
        self._loop.call_soon_threadsafe(self._release.set)

    def close(self) -> None:
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)

    async def _workers(self, request):
        assert request.headers["Authorization"] == f"Bearer {KEY}"
        rows = [
            {"id": w, "engine": "atom", "tp_size": 1, "dp_size": 1}
            for w in self.versions
        ]
        return web.json_response({"protocol_version": 1, "workers": rows})

    async def _join(self, request):
        out = {"success": True, "message": "init_weights_update_group succeeded"}
        return web.json_response({"worker_id": request.match_info["id"], "body": out})

    async def _fanout(self, request):
        assert request.query["selector"] == "engine=atom"
        path = request.match_info["path"]
        body = await request.json() if request.method == "POST" else {}
        self.log.append((path, body, self.in_flight))
        if path == "pause_generation":
            # The engine's pause mode when the body names none.
            assert body.get("mode", "keep") == "keep"
            self.paused = True
        elif path == "continue_generation":
            self.paused = False
        elif path == "receive_weights_rdma":
            for w in self.versions:
                self.versions[w] = str(body["version"])
        results = {
            w: {
                "worker_id": w,
                "status": 200,
                "body": (
                    {"model_path": "m", "weight_version": v}
                    if path == "model_info"
                    else {"success": True, "message": f"{path} succeeded"}
                ),
            }
            for w, v in self.versions.items()
        }
        n = len(results)
        return web.json_response(
            {"results": results, "failed": [], "total": n, "succeeded": n}
        )

    async def _generate(self, request):
        body = await request.json()
        self.in_flight += 1
        await self._release.wait()
        while self.paused:
            await asyncio.sleep(0.01)
        self.in_flight -= 1
        ids = body["input_ids"]
        version = list(self.versions.values())[sum(ids) % len(self.versions)]
        meta = {
            "id": f"g{len(self.log)}",
            "finish_reason": {"type": "stop"},
            "weight_version": version,
            "cached_tokens": 0,
            "output_token_logprobs": [[-0.5, 7, None], [-0.25, 8, None]],
        }
        return web.json_response({"text": "", "output_ids": [7, 8], "meta_info": meta})


class ActorGroup:
    """The actor worker group: its Ray lifecycle as no-ops, and the RCCL
    rendezvous and sender ``AtomMeshRolloutClient`` drives."""

    sent: list[int] = []

    def __init__(self, worker_cls=None, pool=None, num_workers=0, **kwargs) -> None:
        self.num_workers = num_workers

    def start(self) -> None:
        pass

    def call_all(self, *args, **kwargs) -> None:
        pass

    def setup_dispatch_collect_info(self) -> None:
        pass

    def stop(self) -> None:
        pass

    async def _call(self, method, *args):
        if method == "get_rdma_rendezvous":
            return {"address": "127.0.0.1", "port": 29731}
        return True

    def call_single_async(self, idx, method, *args, **kwargs):
        return self._call(method, *args)

    async def _send(self, rank, version, bucket_size_mb):
        if rank == 0:
            ActorGroup.sent.append(version)
        return {"writer": rank == 0, "bytes": 16.0, "buckets": 1}

    def execute_all_async(self, method, **kwargs):
        assert method == "send_weights_rdma"
        return [self._send(r, **kwargs) for r in range(2)]


class FakeCluster:
    def __init__(self, config) -> None:
        pass

    def init(self) -> None:
        pass

    def create_pool(self, name, **kwargs) -> str:
        return name

    def shutdown(self) -> None:
        pass


class FakeTokenizer:
    pad_token = "<pad>"
    pad_token_id = 0
    eos_token = "<eos>"

    def __call__(self, text: str, **kwargs: Any) -> dict:
        return {"input_ids": [ord(c) for c in text]}


def _scores(self, sequences, mask, prompt_lengths, gts, verbose=True):
    """Half the samples right, so dynamic sampling keeps every group."""
    accs = [float(i % 2) for i in range(len(gts))]
    return torch.tensor(accs), [""] * len(gts), accs


@pytest.fixture
def mesh(monkeypatch):
    """A trainer set up through ``_setup_ray_controller`` against the stub,
    with no Ray, model or dataset; yields the stub and the trainer."""
    monkeypatch.setattr(rl_trainer, "RayCluster", FakeCluster)
    monkeypatch.setattr(rl_trainer, "RayWorkerGroup", ActorGroup)
    monkeypatch.setattr(RLTrainer, "_rendezvous_ray_group", lambda self, wg: None)
    monkeypatch.setattr(RLTrainer, "_compute_actor_mp", lambda self: 1)
    monkeypatch.setattr(RLTrainer, "_try_resume_ray_checkpoint", lambda self: None)
    monkeypatch.setattr(RLTrainer, "_compute_rewards_full", _scores)
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setenv("ATOMRL_SKIP_DATASET_INIT", "1")
    ActorGroup.sent = []

    stub = StubMesh()
    cfg = _atomesh(
        f"policy.generation.atom_cfg.api_key={KEY}",
        f"{FULLY_ASYNC}.enabled=true",
        f"{FULLY_ASYNC}.partial_rollout=true",
        f"{FULLY_ASYNC}.staleness_threshold=1.0",
        # Two groups of 8 per step: a period's budget is four groups.
        "policy.train_global_batch_size=16",
        url=stub.url,
    )
    trainer = RLTrainer(cfg)
    trainer._setup_ray_controller()
    try:
        yield stub, trainer
    finally:
        trainer.cleanup()
        stub.close()


def _until(cond, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def _sync_with_groups_in_flight(stub: StubMesh, trainer: RLTrainer):
    """Start the streaming rollout, let a period's groups go out and stay in
    flight, then run the training loop's sync sequence at step 0."""
    assert trainer._resolve_rollout_pipeline(True) == "fully_async"
    pipeline = trainer._start_streaming_rollout(8, 0)
    trainer._rollout_pipeline = pipeline
    _until(lambda: stub.in_flight == 4 * 8)

    trainer.global_step = 0
    pipeline.before_sync()
    trainer._sync_weights_ipc()
    pipeline.after_sync(1)
    return pipeline


def test_a_group_in_flight_across_a_sync_is_accepted_on_the_new_weights(mesh):
    stub, trainer = mesh
    pipeline = _sync_with_groups_in_flight(stub, trainer)

    # The sync paused the workers with every sample in flight, streamed
    # version 1, saw every worker report it, and resumed them.
    assert [(path, held) for path, _, held in stub.log] == [
        ("pause_generation", 32),
        ("receive_weights_rdma", 32),
        ("model_info", 32),
        ("continue_generation", 32),
    ]
    assert stub.log[1][1]["version"] == 1 and ActorGroup.sent == [1]
    assert set(stub.versions.values()) == {"1"}
    assert trainer._atomesh_client.committed_version == 1

    stub.release()
    batch, _, metrics = pipeline.take(1, 1)

    # Both groups were sent under version 0 and answered under version 1.
    assert metrics["async/partial_groups"] == 2.0
    assert metrics["async/staleness_mean"] == 1.0
    sequences = batch[0]
    assert sequences.shape[0] == 16 and sequences[0, -2:].tolist() == [7, 8]
    assert {m["weight_version"] for m in trainer._atomesh_client.sample_meta} == {1}


def test_a_version_the_client_never_committed_still_fails_the_step(mesh):
    """In flight across the sync from 0 to 1, a sample may carry 0 or 1; one
    answered under 2, a sync this client never ran, is refused."""
    stub, trainer = mesh
    pipeline = _sync_with_groups_in_flight(stub, trainer)
    stub.versions = dict.fromkeys(stub.versions, "2")

    stub.release()
    with pytest.raises(RuntimeError, match="streaming rollout failed") as err:
        pipeline.take(1, 1)
    assert isinstance(err.value.__cause__, WeightVersionMismatch)
