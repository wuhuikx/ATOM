# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``/receive_weights_rdma`` under ``--enable-rl``: one weight stream through
the engine's receive command, whose commit makes ``str(version)`` the version
``/model_info`` and ``/generate`` report.

The core manager and the engines' utility handlers are real; only the TP
runners are mocked."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import test_kv_change
from import_guard import skip_if_dependency_missing
from test_kv_change import _running, _wired

try:
    from fastapi.testclient import TestClient

    from atom.entrypoints.openai import api_server
except ImportError as exc:  # pragma: no cover - environment-dependent skip
    skip_if_dependency_missing(exc, "api_server import unavailable")

from atom.model_engine.collective_rpc import (
    DISCARD_WEIGHT_SYNC_CMD,
    RECEIVE_WEIGHTS_RDMA_CMD,
    RpcResult,
)
from atom.model_engine.engine_utility import WEIGHT_STREAM_TIMEOUT_S
from atom.model_engine.scheduler import ScheduledBatchOutput
from atom.model_engine.sequence import Sequence
from atom.sampling_params import SamplingParams

STREAM = {"group_name": "atomesh-rl-weights-4121", "version": 7}


class _Engine:
    """The API server's engine over real schedulers and a wired core manager.

    ``preprocess`` runs the prompt through DP rank 0's scheduler for one
    token, so ``/generate`` reports what that scheduler stamps.
    """

    def __init__(self, schedulers, runners=None):
        self.schedulers = schedulers
        self.core_mgr, self.engines = _wired(schedulers, runners)
        self.config = SimpleNamespace(model="/ckpt/base", max_model_len=None)
        self.io_processor = self
        self.requests = {}
        self.timeouts = []
        broadcast = self.core_mgr.broadcast_utility_command_sync

        def recording_broadcast(cmd, timeout=300.0, **kwargs):
            self.timeouts.append((cmd, timeout))
            return broadcast(cmd, timeout=timeout, **kwargs)

        self.core_mgr.broadcast_utility_command_sync = recording_broadcast
        # /generate's submission: preprocess has already run the request.
        self.core_mgr.add_request = lambda seqs: None
        self.core_mgr.abort_request = lambda seq_id: None

    def preprocess(self, prompt, sampling_params, stream_callback=None, **_kwargs):
        sched = self.schedulers[0]
        seq = Sequence(list(prompt), 4, SamplingParams(max_tokens=1, ignore_eos=True))
        self.requests[seq.id] = seq
        sched.add(seq)
        sched.schedule()
        queue = mock.Mock()
        sched.postprocess(
            [seq],
            ScheduledBatchOutput([seq.id], [(11,)], None, None, None),
            stream_output_queue=queue,
        )
        for _, output in queue.put_nowait.call_args.args[0]:
            stream_callback(output)
        return seq


@pytest.fixture
def serve(monkeypatch):
    """Register the RL routes for *enable_rl* over an ``_Engine``; the app's
    routes are put back afterwards."""
    routes = list(api_server.app.router.routes)
    engine_routes = set(api_server._ENGINE_ROUTES)
    monkeypatch.setattr(api_server, "delivered_text", lambda ids: f"text{list(ids)}")
    monkeypatch.setattr(api_server, "record_nonstream_first_token", lambda: None)

    def start(dp=1, runners=None, enable_rl=True):
        fake = _Engine([_running(0)[0] for _ in range(dp)], runners)
        monkeypatch.setattr(api_server, "engine", fake)
        api_server.register_rl_routes(SimpleNamespace(enable_rl=enable_rl))
        return TestClient(api_server.app), fake

    yield start
    api_server.app.router.routes[:] = routes
    api_server._ENGINE_ROUTES.clear()
    api_server._ENGINE_ROUTES.update(engine_routes)
    api_server.app.openapi_schema = None


def _reported_versions(client):
    info = client.get("/model_info")
    generated = client.post("/generate", json={"input_ids": [1, 2, 3]})
    assert info.status_code == 200 and generated.status_code == 200
    return (
        info.json()["weight_version"],
        generated.json()["meta_info"]["weight_version"],
    )


def test_a_committed_stream_advances_the_weight_version(serve):
    client, fake = serve(dp=2)
    assert _reported_versions(client) == ("default", "default")

    reply = client.post("/receive_weights_rdma", json=STREAM)

    assert reply.status_code == 200
    assert reply.json() == {
        "success": True,
        "message": "receive_weights_rdma succeeded",
    }
    assert _reported_versions(client) == ("7", "7")
    assert [s.weight_version for s in fake.schedulers] == ["7", "7"]


def test_the_stream_runs_the_engines_receive_command(serve):
    runner = test_kv_change._Runners()
    client, fake = serve(runners=[runner])

    assert client.post("/receive_weights_rdma", json=STREAM).status_code == 200

    ((method, payload),) = runner.calls
    assert method == RECEIVE_WEIGHTS_RDMA_CMD
    assert payload.args == ("atomesh-rl-weights-4121", 7)
    # It waits as long as the engines do, as AsyncLLMEngine's call does.
    assert (RECEIVE_WEIGHTS_RDMA_CMD, WEIGHT_STREAM_TIMEOUT_S) in fake.timeouts


def test_the_next_stream_advances_it_again(serve):
    client, fake = serve()
    assert client.post("/receive_weights_rdma", json=STREAM).status_code == 200
    reply = client.post("/receive_weights_rdma", json={**STREAM, "version": 8})
    assert reply.status_code == 200
    assert _reported_versions(client) == ("8", "8")


def test_a_failed_stream_leaves_the_old_version(serve, monkeypatch):
    """One engine's ranks fail mid-stream: no FINISH, so neither engine
    commits, and both stay fenced over what they wrote."""
    client, fake = serve(dp=2)
    calls = []

    def utility_rpc(self, method, payload, timeout=300.0):
        calls.append(method)
        if method == RECEIVE_WEIGHTS_RDMA_CMD and calls.count(method) == 2:
            return [RpcResult(payload.request_id, 0, error="stream broke at bucket 3")]
        return [RpcResult(payload.request_id, 0, value=1)]

    monkeypatch.setattr(test_kv_change._Runners, "utility_rpc", utility_rpc)

    reply = client.post("/receive_weights_rdma", json=STREAM)

    assert reply.status_code == 400 and reply.json()["success"] is False
    assert "stream broke at bucket 3" in reply.json()["message"]
    assert DISCARD_WEIGHT_SYNC_CMD in calls
    assert [s.weight_version for s in fake.schedulers] == ["default", "default"]
    assert all(e._weight_sync_fenced for e in fake.engines)
    assert _reported_versions(client) == ("default", "default")


def test_a_stream_one_tp_rank_fails_leaves_the_old_version(serve):
    """Rank 1 wrote what rank 0 failed, so the engine's weights are partial."""
    runner = test_kv_change._Runners(
        unasked=True, refusal="rank 0 lost the stream", refusing={0}, ranks=2
    )
    client, fake = serve(runners=[runner])

    reply = client.post("/receive_weights_rdma", json=STREAM)

    assert reply.status_code == 400
    assert "rank 0 lost the stream" in reply.json()["message"]
    assert fake.engines[0]._weight_sync_fenced
    assert _reported_versions(client) == ("default", "default")


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"version": 7},
        {"group_name": "", "version": 7},
        {"group_name": 3, "version": 7},
        {"group_name": "g"},
        {"group_name": "g", "version": "7"},
        {"group_name": "g", "version": 7.0},
        {"group_name": "g", "version": True},
    ],
)
def test_a_body_without_a_group_and_integer_version_is_refused(serve, body):
    client, fake = serve()
    reply = client.post("/receive_weights_rdma", json=body)
    assert reply.status_code == 400 and reply.json()["success"] is False
    assert fake.timeouts == []
    assert fake.schedulers[0].weight_version == "default"


def test_without_enable_rl_the_route_does_not_exist(serve):
    client, fake = serve(enable_rl=False)
    assert client.post("/receive_weights_rdma", json=STREAM).status_code == 404
    assert "/receive_weights_rdma" not in api_server._ENGINE_ROUTES
    assert fake.timeouts == []


def test_a_frontend_worker_relays_the_route_to_the_hub(serve):
    serve()
    assert "/receive_weights_rdma" in api_server._ENGINE_ROUTES
