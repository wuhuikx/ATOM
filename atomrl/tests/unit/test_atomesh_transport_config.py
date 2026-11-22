"""``atom_cfg.transport: atomesh``: what the config accepts, the two 4+4 DAPO
configs derived from the ATOM BF16 long run, and the trainer's rollout and
weight sync going through the ATOMesh client (mocked) instead of Ray replicas."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from atomrl.core.config import AtomRLConfig
from atomrl.engine.inference import atomesh_rollout
from atomrl.trainer import rl_trainer
from atomrl.trainer.rl_trainer import RLTrainer
from omegaconf import OmegaConf

CONFIGS = Path(__file__).resolve().parents[2] / "examples" / "DAPO" / "configs"
LONGRUN = CONFIGS / "dapo_qwen3_8b_ray_atom_bf16_longrun.yaml"
DIRECT = CONFIGS / "dapo_qwen3_8b_atom_bf16_4p4_direct.yaml"
ATOMESH = CONFIGS / "dapo_qwen3_8b_atom_bf16_4p4_atomesh.yaml"
MESH_URL = "http://127.0.0.1:18400"


@pytest.fixture(autouse=True)
def _data_root(monkeypatch):
    monkeypatch.setenv("DATA_ROOT", "/data")


def _load(path: Path, *overrides: str) -> AtomRLConfig:
    return AtomRLConfig.from_yaml(path, overrides=list(overrides))


def _atomesh(*overrides: str) -> AtomRLConfig:
    return _load(ATOMESH, f"policy.generation.atom_cfg.base_url={MESH_URL}", *overrides)


# ── config validation ──────────────────────────────────────────────────────


def test_the_atomesh_config_is_accepted():
    cfg = _atomesh("policy.generation.atom_cfg.api_key=k")
    atom = cfg.policy.generation.atom_cfg
    assert (atom.transport, atom.base_url, atom.api_key) == ("atomesh", MESH_URL, "k")
    assert cfg.weight_sync.backend == "rdma"
    assert cfg.weight_sync.rdma.require_rdma is False


def test_the_direct_config_is_accepted():
    cfg = _load(DIRECT)
    assert cfg.policy.generation.atom_cfg.transport == "ray_http"
    assert cfg.controller.ray.rollout.num_workers == 4


@pytest.mark.parametrize(
    ("override", "named"),
    [
        # colocation: CUDA IPC or a shared folder to replicas on the trainer's
        # GPUs, sleeping between rollout and training, a Ray rollout pool
        ("weight_sync.backend=auto", "weight_sync.backend"),
        ("weight_sync.backend=shared_folder", "weight_sync.backend"),
        ("policy.generation.vllm_cfg.enable_sleep_mode=true", "enable_sleep_mode"),
        ("controller.ray.rollout.num_workers=4", "rollout.num_workers"),
        # what the workers' weight stream cannot do
        ("weight_sync.rdma.require_rdma=true", "require_rdma"),
        ("weight_sync.fp8_quantize=true", "FP8"),
        ("weight_sync.fp8_quantization_location=trainer", "FP8"),
        # what the trainer cannot drive through ATOMesh
        ("policy.generation.atom_cfg.base_url=''", "base_url"),
        ("policy.generation_backend=vllm", "generation_backend"),
        ("controller.ray.enabled=false", "controller.ray.enabled"),
        ("controller.ray.one_step_off_policy=true", "one_step_off_policy"),
        ("controller.ray.fully_async.enabled=true", "fully_async"),
    ],
)
def test_atomesh_refuses(override, named):
    with pytest.raises(ValueError, match="transport='atomesh'") as err:
        _atomesh(override)
    assert named in str(err.value)


def test_every_refusal_is_named_at_once():
    with pytest.raises(ValueError) as err:
        _atomesh("weight_sync.backend=auto", "controller.ray.rollout.num_workers=4")
    assert "weight_sync.backend" in str(err.value)
    assert "rollout.num_workers" in str(err.value)


def test_the_colocated_longrun_is_untouched_by_the_check():
    """The same knobs stay legal on the other transports."""
    cfg = _load(LONGRUN, "policy.generation.vllm_cfg.enable_sleep_mode=true")
    assert cfg.weight_sync.backend == "auto"


# ── the derived configs ────────────────────────────────────────────────────


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


# Each 4+4 config differs from the long run in these keys only: topology, the
# weight-sync backend, the step count, LLM-Booster#718's values, and the run's
# own name and checkpoint directory.
TOPOLOGY = {"controller.ray.actor.num_workers"}
WEIGHT_SYNC = {"weight_sync.backend", "weight_sync.rdma.require_rdma"}
STEPS = {"num_training_steps"}
FIXED_BY_718 = {
    "algorithm.dapo.num_generations",
    "policy.max_response_length",
    "algorithm.dapo.max_resp_len",
    "policy.generation.atom_cfg.enable_prefix_caching",
}
RUN_IDENTITY = {"checkpointing.checkpoint_dir", "logger.wandb.name"}
COMMON = TOPOLOGY | WEIGHT_SYNC | STEPS | FIXED_BY_718 | RUN_IDENTITY


@pytest.mark.parametrize(
    ("path", "own"),
    [
        (DIRECT, {"controller.ray.rollout.num_workers"}),
        (
            ATOMESH,
            {
                "cluster.gpus_per_node",
                "policy.generation.atom_cfg.transport",
                "policy.generation.atom_cfg.base_url",
                "policy.generation.atom_cfg.api_key",
            },
        ),
    ],
)
def test_the_4p4_configs_differ_from_the_longrun_only_where_allowed(path, own):
    base, derived = _flat(LONGRUN), _flat(path)
    changed = {k for k in base.keys() | derived.keys() if base.get(k) != derived.get(k)}
    assert changed == COMMON | own


def test_the_4p4_configs_carry_718s_values():
    for path in (DIRECT, ATOMESH):
        flat = _flat(path)
        assert flat["algorithm.dapo.num_generations"] == 8
        assert flat["policy.max_response_length"] == 4096
        assert flat["algorithm.dapo.max_resp_len"] == 4096
        assert flat["policy.generation.atom_cfg.enable_prefix_caching"] is True
        assert flat["num_training_steps"] == 30
        assert flat["controller.ray.actor.num_workers"] == 4
        assert flat["weight_sync.backend"] == "rdma"
        assert flat["weight_sync.rdma.require_rdma"] is False


# ── the trainer under atomesh, with the client mocked ──────────────────────


class FakeClient:
    """``AtomMeshRolloutClient``'s surface, recording what the trainer calls."""

    instances: list[FakeClient] = []

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        self.base_url = base_url
        self.kwargs = kwargs
        self.calls: list[tuple] = []
        self.worker_ids = ["w0", "w1", "w2", "w3"]
        FakeClient.instances.append(self)

    async def join_weight_group(self, trainer, **kwargs: Any) -> dict:
        self.calls.append(("join", trainer, kwargs))
        return {"master_addr": "10.0.0.5", "master_port": 29731, "world_size": 5}

    async def generate_batch(self, prompts, sampling) -> list[dict]:
        self.calls.append(("generate", [list(p) for p in prompts], dict(sampling)))
        return [
            {
                "text": "",
                "prompt_token_ids": list(p),
                "token_ids": [7, 8],
                "logprobs": [-0.5, -0.25],
            }
            for p in prompts
        ]

    async def sync_weights(self, version: int, *, bucket_size_mb: int) -> dict:
        self.calls.append(("sync", version, bucket_size_mb))
        return {"writer": True, "bytes": 16e9, "buckets": 8, "bcast_s": 1.5}

    async def leave_weight_group(self) -> None:
        self.calls.append(("leave",))


class FakeWorkerGroup:
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


class ReplicaManagerCalled(AssertionError):
    pass


def _no_replicas(*args, **kwargs):
    raise ReplicaManagerCalled("the direct path's ATOM replicas were built")


@pytest.fixture
def ray_free(monkeypatch):
    """``_setup_ray_controller`` without Ray, a model or a dataset."""
    FakeClient.instances.clear()
    monkeypatch.setattr(atomesh_rollout, "AtomMeshRolloutClient", FakeClient)
    monkeypatch.setattr(rl_trainer, "RayCluster", FakeCluster)
    monkeypatch.setattr(rl_trainer, "RayWorkerGroup", FakeWorkerGroup)
    monkeypatch.setattr(RLTrainer, "_rendezvous_ray_group", lambda self, wg: None)
    monkeypatch.setattr(RLTrainer, "_compute_actor_mp", lambda self: 1)
    monkeypatch.setattr(RLTrainer, "_try_resume_ray_checkpoint", lambda self: None)
    monkeypatch.setattr(
        RLTrainer, "_release_actor_cached_memory", lambda self: _no_replicas()
    )
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = types.SimpleNamespace(
        from_pretrained=lambda *a, **k: FakeTokenizer()
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setenv("ATOMRL_SKIP_DATASET_INIT", "1")


def _setup(cfg: AtomRLConfig) -> RLTrainer:
    trainer = RLTrainer(cfg)
    trainer._setup_ray_controller()
    return trainer


def test_the_trainer_builds_the_atomesh_client_and_joins_once(ray_free):
    cfg = _atomesh(
        "policy.generation.atom_cfg.api_key=k", "weight_sync.rdma.interface=lo"
    )
    trainer = _setup(cfg)

    [client] = FakeClient.instances
    assert client.base_url == MESH_URL
    assert client.kwargs["api_key"] == "k"
    assert client.kwargs["control_timeout_s"] == float(cfg.weight_sync.timeout_s)
    assert client.kwargs["max_model_len"] == 21504
    [(call, wg, kwargs)] = client.calls
    assert call == "join" and wg is trainer._actor_wg
    assert kwargs["interface"] == "lo"
    assert kwargs["timeout_s"] == cfg.weight_sync.timeout_s
    assert trainer._ray_use_atom and trainer._ray_rollout_mgr is None
    assert trainer._atom_engine is None


def test_rollout_goes_through_the_client(ray_free):
    trainer = _setup(_atomesh())
    [client] = FakeClient.instances

    seqs, mask, plen, lp, _ = trainer._rollout_with_ray_vllm(["ab", "c"], 2)

    [(_, prompts, sampling)] = [c for c in client.calls if c[0] == "generate"]
    assert prompts == [[97, 98], [97, 98], [99], [99]]
    assert sampling == {
        "max_tokens": 4096,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "logprobs": 0,
    }
    assert plen == [2, 2, 1, 1]
    assert seqs[0].tolist() == [97, 98, 7, 8]
    assert seqs[2].tolist() == [0, 99, 7, 8]
    assert lp is not None and lp[0, -2:].tolist() == [-0.5, -0.25]


def test_weight_sync_goes_through_the_client(ray_free):
    cfg = _atomesh("weight_sync.bucket_size_mb=512")
    trainer = _setup(cfg)
    [client] = FakeClient.instances
    trainer.global_step = 6

    trainer._sync_weights_ipc()

    assert client.calls[-1] == ("sync", 7, 512)
    metrics = trainer._last_weight_sync_metrics
    assert metrics["weight_sync/backend_atomesh"] == 1.0
    assert metrics["weight_sync/version"] == 7.0
    assert metrics["weight_sync/bytes"] == 16e9
    assert metrics["timing/weight_sync_rdma_sender_bcast_s"] == 1.5
    assert metrics["timing/weight_sync_rdma_s"] >= 0.0


def test_the_engine_never_sleeps_and_cleanup_leaves_the_group(ray_free):
    trainer = _setup(_atomesh())
    [client] = FakeClient.instances
    engine = trainer._ray_vllm_engine
    engine.sleep()
    engine.wake(tags=["weights"])
    assert not engine.enable_sleep and not engine._sleeping
    assert [c[0] for c in client.calls] == ["join"]

    trainer.cleanup()

    assert client.calls[-1] == ("leave",)


def test_ray_http_still_builds_the_direct_replicas(ray_free, monkeypatch):
    """The direct config takes the replica path, and no ATOMesh client."""
    with pytest.raises(ReplicaManagerCalled):
        _setup(_load(DIRECT))
    assert FakeClient.instances == []
