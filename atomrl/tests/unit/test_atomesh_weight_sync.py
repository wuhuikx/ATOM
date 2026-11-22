"""AtomMeshRolloutClient's weight group and weight sync against a stub ATOMesh
``/v1/rl`` and a stub trainer worker group."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from aiohttp import web
from atomrl.engine.inference.atomesh_rollout import (
    AtomMeshRolloutClient,
    WeightSyncError,
    WeightVersionMismatch,
)

KEY = "rl-secret"
GROUP = "atomesh-rl-weights-test"


class Worker:
    def __init__(self, wid: str, tp_size: int):
        self.id = wid
        self.tp_size = tp_size
        self.weight_version = "0"
        self.paused = False
        self.joined: dict | None = None
        # Failure switches a test sets.
        self.refuse: set[str] = set()
        self.stale_after_receive = False


class Mesh:
    """ATOMesh's ``/v1/rl`` routes over in-process workers, recording every
    call, plus ``/generate`` answered by the first worker."""

    def __init__(self, tp_sizes=(1, 2, 1)):
        self.workers = [Worker(f"w{i}", tp) for i, tp in enumerate(tp_sizes)]
        self.log: list[tuple] = []
        self.auth: list[str | None] = []
        # Set when the trainer's sender runs: a receive completes only once
        # the broadcast it waits for has started.
        self.broadcast = asyncio.Event()
        self.deregistered: set[str] = set()

    def _engine_call(self, w: Worker, path: str, body: dict) -> tuple[int, dict]:
        if path in w.refuse:
            return 400, {"success": False, "message": f"{path} refused"}
        if path == "init_weights_update_group":
            w.joined = body
        elif path == "pause_generation":
            w.paused = True
        elif path == "continue_generation":
            w.paused = False
        elif path == "receive_weights_rdma":
            if not w.stale_after_receive:
                w.weight_version = str(body["version"])
        elif path == "model_info":
            return 200, {"model_path": "m", "weight_version": w.weight_version}
        return 200, {"success": True, "message": f"{path} succeeded"}

    async def _workers(self, request):
        self.auth.append(request.headers.get("Authorization"))
        self.log.append(("GET", "workers"))
        rows = [
            {"id": w.id, "engine": "atom", "tp_size": w.tp_size, "dp_size": 1}
            for w in self.workers
        ]
        rows.append({"id": "s0", "engine": "sglang", "tp_size": 8, "dp_size": 1})
        return web.json_response({"protocol_version": 1, "workers": rows})

    async def _proxy(self, request):
        self.auth.append(request.headers.get("Authorization"))
        w = next(w for w in self.workers if w.id == request.match_info["id"])
        path = request.match_info["path"]
        body = await request.json()
        self.log.append(("POST", path, w.id))
        status, out = self._engine_call(w, path, body)
        return web.json_response(
            {"worker_id": w.id, "url": w.id, "status": status, "body": out},
            status=status,
        )

    async def _fanout(self, request):
        self.auth.append(request.headers.get("Authorization"))
        assert request.query["selector"] == "engine=atom"
        path = request.match_info["path"]
        body = await request.json() if request.method == "POST" else {}
        self.log.append((request.method, path, body))
        if path == "receive_weights_rdma":
            await self.broadcast.wait()
        results, failed = {}, []
        for w in self.workers:
            if w.id in self.deregistered:
                continue
            status, out = self._engine_call(w, path, body)
            results[w.id] = {"worker_id": w.id, "status": status, "body": out}
            if status != 200:
                failed.append(
                    {"worker_id": w.id, "error": "upstream_error", "status": status}
                )
        report = {
            "results": results,
            "failed": failed,
            "total": len(results),
            "succeeded": len(results) - len(failed),
        }
        return web.json_response(report, status=207 if failed else 200)

    async def _generate(self, request):
        body = await request.json()
        meta = {
            "id": "g",
            "finish_reason": {"type": "stop"},
            "weight_version": self.workers[0].weight_version,
        }
        return web.json_response({"text": "", "output_ids": [1], "meta_info": meta})

    async def run(self, use):
        app = web.Application()
        app.router.add_get("/v1/rl/workers", self._workers)
        app.router.add_post("/v1/rl/workers/{id}/engine/{path}", self._proxy)
        app.router.add_route("*", "/v1/rl/engine/{path}", self._fanout)
        app.router.add_post("/generate", self._generate)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await asyncio.wait_for(use(f"http://127.0.0.1:{port}"), 30)
        finally:
            await runner.cleanup()


class Trainer:
    """The actor worker group's async calls, as awaitables."""

    def __init__(self, mesh: Mesh, ranks: int = 2, fail_send: bool = False):
        self.mesh = mesh
        self.ranks = ranks
        self.fail_send = fail_send

    async def _call(self, method, *args, **kwargs):
        self.mesh.log.append(("trainer", method, args, kwargs))
        if method == "get_rdma_rendezvous":
            return {"address": "10.0.0.5", "port": 29731}
        return True

    def call_single_async(self, idx, method, *args, **kwargs):
        assert idx == 0
        return self._call(method, *args, **kwargs)

    async def _send(self, rank, version, bucket_size_mb):
        self.mesh.broadcast.set()
        if rank == 0:
            self.mesh.log.append(("trainer", "send_weights_rdma", version))
            if self.fail_send:
                raise RuntimeError("NCCL broadcast failed")
            return {"writer": True, "bytes": 16.0, "buckets": 1}
        return {"writer": False}

    def execute_all_async(self, method, **kwargs):
        assert method == "send_weights_rdma"
        return [self._send(r, **kwargs) for r in range(self.ranks)]


def _run(mesh: Mesh, body, trainer: Trainer | None = None) -> Any:
    trainer = trainer or Trainer(mesh)

    async def use(url):
        client = AtomMeshRolloutClient(url, api_key=KEY)
        await client.join_weight_group(trainer, interface="eth0", group_name=GROUP)
        mesh.log.clear()
        return await body(client)

    return asyncio.run(mesh.run(use))


def test_join_places_workers_by_tp_size_once():
    mesh = Mesh(tp_sizes=(1, 2, 1))
    trainer = Trainer(mesh)

    async def use(url):
        client = AtomMeshRolloutClient(url, api_key=KEY)
        info = await client.join_weight_group(
            trainer, interface="eth0", timeout_s=600, group_name=GROUP
        )
        with pytest.raises(WeightSyncError, match="already joined"):
            await client.join_weight_group(trainer, interface="eth0")
        return client, info

    client, info = asyncio.run(mesh.run(use))
    assert info == {"master_addr": "10.0.0.5", "master_port": 29731, "world_size": 5}
    assert [w.joined["rank_offset"] for w in mesh.workers] == [1, 2, 4]
    assert mesh.workers[0].joined == {
        "master_address": "10.0.0.5",
        "master_port": 29731,
        "rank_offset": 1,
        "world_size": 5,
        "group_name": GROUP,
        "backend": "nccl",
    }
    assert (
        "trainer",
        "init_rdma_weight_group",
        ("10.0.0.5", 29731, 5, GROUP, 600),
        {},
    ) in mesh.log
    assert client.worker_ids == ["w0", "w1", "w2"]
    assert set(mesh.auth) == {f"Bearer {KEY}"}


def test_a_worker_refusing_to_join_raises():
    mesh = Mesh()
    mesh.workers[1].refuse.add("init_weights_update_group")

    async def use(url):
        client = AtomMeshRolloutClient(url, api_key=KEY)
        with pytest.raises(WeightSyncError, match="w1 did not join"):
            await client.join_weight_group(Trainer(mesh), interface="eth0")
        return client

    assert asyncio.run(mesh.run(use)).group_name is None


def test_sync_call_sequence_matches_the_contract():
    mesh = Mesh()

    async def body(client):
        stats = await client.sync_weights(7, bucket_size_mb=512)
        return client, stats

    client, stats = _run(mesh, body)
    assert stats["writer"] is True
    sequence = [entry[:2] for entry in mesh.log]
    # The receive and the send run together (the stub's receive answers only
    # once the broadcast has started), so their relative order is free.
    assert sequence[0] == ("POST", "pause_generation")
    assert set(sequence[1:3]) == {
        ("POST", "receive_weights_rdma"),
        ("trainer", "send_weights_rdma"),
    }
    assert sequence[3:] == [("GET", "model_info"), ("POST", "continue_generation")]
    assert (
        "POST",
        "receive_weights_rdma",
        {"group_name": GROUP, "version": 7},
    ) in mesh.log
    assert ("trainer", "send_weights_rdma", 7) in mesh.log
    # No rendezvous or join during the sync: the group was formed at setup.
    assert not any("init" in str(e[1]) for e in mesh.log)
    assert client.committed_version == 7
    assert [w.weight_version for w in mesh.workers] == ["7", "7", "7"]
    assert not any(w.paused for w in mesh.workers)
    assert set(mesh.auth) == {f"Bearer {KEY}"}


def test_generation_after_a_sync_accepts_only_the_new_version():
    mesh = Mesh()

    async def body(client):
        await client.sync_weights(7, bucket_size_mb=512)
        rows = await client.generate_batch([[5]], {"max_tokens": 4})
        mesh.workers[0].weight_version = "6"
        with pytest.raises(WeightVersionMismatch):
            await client.generate_batch([[5]], {"max_tokens": 4})
        return rows

    assert _run(mesh, body)[0]["token_ids"] == [1]


def test_a_failed_receive_leaves_workers_resumed_and_raises():
    mesh = Mesh()
    mesh.workers[2].refuse.add("receive_weights_rdma")

    async def body(client):
        with pytest.raises(
            WeightSyncError, match="receive_weights_rdma reached 2 of 3"
        ):
            await client.sync_weights(7, bucket_size_mb=512)
        return client

    client = _run(mesh, body)
    assert mesh.log[-1][:2] == ("POST", "continue_generation")
    assert not any(w.paused for w in mesh.workers)
    assert client.committed_version == 0
    assert ("GET", "model_info", {}) not in mesh.log


def test_a_failed_sender_leaves_workers_resumed_and_raises():
    mesh = Mesh()

    async def body(client):
        with pytest.raises(WeightSyncError, match="NCCL broadcast failed"):
            await client.sync_weights(7, bucket_size_mb=512)
        return client

    client = _run(mesh, body, Trainer(mesh, fail_send=True))
    assert mesh.log[-1][:2] == ("POST", "continue_generation")
    assert not any(w.paused for w in mesh.workers)
    assert client.committed_version == 0


def test_a_worker_that_misses_the_version_is_reported():
    mesh = Mesh()
    mesh.workers[1].stale_after_receive = True

    async def body(client):
        with pytest.raises(WeightSyncError) as raised:
            await client.sync_weights(7, bucket_size_mb=512)
        return client, str(raised.value)

    client, message = _run(mesh, body)
    assert "{'w1': '0'}" in message
    assert client.committed_version == 0
    assert not any(w.paused for w in mesh.workers)


def test_a_worker_gone_since_setup_fails_the_sync():
    mesh = Mesh()

    async def body(client):
        mesh.deregistered.add("w2")
        with pytest.raises(WeightSyncError, match="reached 2 of 3"):
            await client.sync_weights(7, bucket_size_mb=512)
        return client

    assert _run(mesh, body).committed_version == 0
    assert mesh.log[-1][:2] == ("POST", "continue_generation")


def test_paused_resumes_when_its_body_raises():
    mesh = Mesh()

    async def body(client):
        with pytest.raises(ValueError):
            async with client.paused():
                assert all(w.paused for w in mesh.workers)
                raise ValueError("boom")

    _run(mesh, body)
    assert [e[:2] for e in mesh.log] == [
        ("POST", "pause_generation"),
        ("POST", "continue_generation"),
    ]
    assert not any(w.paused for w in mesh.workers)


def test_sync_before_join_raises():
    async def use(url):
        with pytest.raises(WeightSyncError, match="join_weight_group"):
            await AtomMeshRolloutClient(url).sync_weights(1, bucket_size_mb=1)

    asyncio.run(Mesh().run(use))


def test_leave_destroys_the_group_on_workers_and_trainer():
    mesh = Mesh()

    async def body(client):
        await client.leave_weight_group()
        return client

    client = _run(mesh, body)
    assert mesh.log == [
        ("POST", "destroy_weights_update_group", {"group_name": GROUP}),
        ("trainer", "destroy_rdma_weight_group", (), {}),
    ]
    assert client.group_name is None
