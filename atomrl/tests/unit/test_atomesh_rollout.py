"""AtomMeshRolloutClient against a stub ATOMesh ``/generate``."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import pytest
from aiohttp import web
from atomrl.engine.inference.atom_ray_server import ATOMRayServer
from atomrl.engine.inference.atomesh_rollout import (
    AtomMeshRolloutClient,
    AtomMeshRolloutError,
    WeightVersionMismatch,
)

# What the engine produces for each prompt, keyed by the prompt's first id.
OUTPUTS = {
    151644: {"text": " 42", "token_ids": [220, 19, 17, 151643]},
    9906: {"text": "Hi", "token_ids": [13347]},
    77: {"text": " no.", "token_ids": [902, 13, 151645]},
}


def _logprobs(token_ids: list[int]) -> list[float]:
    return [-0.125 * (i + 1) for i in range(len(token_ids))]


def _generate_body(body: dict[str, Any], weight_version: str = "0") -> dict:
    """What an ATOM worker answers through ATOMesh for ``body``."""
    out = OUTPUTS[body["input_ids"][0]]
    meta = {
        "id": f"generate-{len(out['token_ids'])}",
        "finish_reason": {"type": "stop"},
        "prompt_tokens": len(body["input_ids"]),
        "completion_tokens": len(out["token_ids"]),
        "cached_tokens": 3,
        "weight_version": weight_version,
    }
    if body.get("return_logprob"):
        meta["output_token_logprobs"] = [
            [lp, t, None]
            for lp, t in zip(_logprobs(out["token_ids"]), out["token_ids"])
        ]
    return {"text": out["text"], "output_ids": out["token_ids"], "meta_info": meta}


class Stub:
    """A stub ``/generate``; ``answer(body)`` gives ``(status, json)``."""

    def __init__(self, answer: Callable[[dict], tuple[int, Any]] | None = None):
        self.answer = answer or (lambda body: (200, _generate_body(body)))
        self.bodies: list[dict] = []

    async def _handle(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.bodies.append(body)
        status, payload = self.answer(body)
        return web.json_response(payload, status=status)

    async def run(self, use: Callable[[str], Any]) -> Any:
        app = web.Application()
        app.router.add_post("/generate", self._handle)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        try:
            return await use(f"http://127.0.0.1:{port}")
        finally:
            await runner.cleanup()


def _generate(stub: Stub, prompts, sampling, **client_kwargs):
    async def use(url):
        client = AtomMeshRolloutClient(url, **client_kwargs)
        return client, await client.generate_batch(prompts, sampling)

    return asyncio.run(stub.run(use))


class FakeEngine:
    """ATOM's ``LLMEngine.generate`` on the same outputs the stub serves."""

    def generate(self, prompts, params, request_ids=None):
        outs = []
        for prompt, sp in zip(prompts, params):
            out = OUTPUTS[prompt[0]]
            for _ in range(sp.n):
                outs.append(
                    {
                        "text": out["text"],
                        "token_ids": list(out["token_ids"]),
                        "logprobs": (
                            _logprobs(out["token_ids"]) if sp.logprobs else None
                        ),
                    }
                )
        return outs

    def get_metrics_statistics(self):
        return {}


def _direct(prompts, sampling):
    server = ATOMRayServer("m", {"max_model_len": 4096}, replica_rank=0)
    server.engine = FakeEngine()
    return asyncio.run(server.generate_batch(prompts, sampling))


PROMPTS = [[151644, 872, 198, 785]] * 3 + [[9906, 11]] * 3 + [[77]] * 3


@pytest.mark.parametrize("want_logprobs", [True, False])
def test_rows_match_atom_ray_server_generate_batch(want_logprobs):
    sampling = {"max_tokens": 64, "temperature": 1.0, "top_p": 1.0, "top_k": -1}
    if want_logprobs:
        sampling["logprobs"] = 0

    direct = _direct(PROMPTS, sampling)
    _, mesh = _generate(Stub(), PROMPTS, sampling, max_model_len=4096)

    assert mesh == direct
    assert [type(v) for row in mesh for v in row.values()] == [
        type(v) for row in direct for v in row.values()
    ]
    assert (mesh[0]["logprobs"] is None) is (not want_logprobs)


def test_a_group_of_eight_is_eight_requests():
    stub = Stub()
    prompt = [151644, 872, 198, 785]

    async def use(url):
        return await AtomMeshRolloutClient(url).generate_group(
            prompt, 8, {"max_tokens": 16}
        )

    rows = asyncio.run(stub.run(use))

    assert len(stub.bodies) == 8
    assert all(b == stub.bodies[0] for b in stub.bodies)
    assert stub.bodies[0]["input_ids"] == prompt
    assert len(rows) == 8
    assert all(r["prompt_token_ids"] == prompt for r in rows)


def test_every_sampling_field_is_sent():
    stub = Stub()
    sampling = {
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "logprobs": 0,
    }
    _generate(stub, [[151644, 872, 198, 785]], sampling, max_model_len=1000)

    assert stub.bodies == [
        {
            "input_ids": [151644, 872, 198, 785],
            # Clamped to what the context leaves, as the direct path does.
            "sampling_params": {
                "max_new_tokens": 996,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 20,
            },
            "return_logprob": True,
        }
    ]


def test_an_unknown_sampling_key_is_refused_before_sending():
    stub = Stub()
    with pytest.raises(ValueError, match="seed"):
        _generate(stub, [[77]], {"max_tokens": 8, "seed": 1})
    assert stub.bodies == []


def test_a_sample_from_an_uncommitted_version_is_rejected():
    # A worker that re-registered fresh answers "0" to a client at 6.
    stub = Stub(lambda body: (200, _generate_body(body, weight_version="0")))
    with pytest.raises(WeightVersionMismatch) as err:
        _generate(stub, [[77]], {"max_tokens": 8}, committed_version=6)
    assert (err.value.version, err.value.low, err.value.high) == ("0", 6, 6)
    assert err.value.request_id == "generate-3"


def test_a_version_newer_than_committed_is_rejected():
    stub = Stub(lambda body: (200, _generate_body(body, weight_version="7")))
    with pytest.raises(WeightVersionMismatch):
        _generate(stub, [[77]], {"max_tokens": 8}, committed_version=6)


# A: sent at 6, finished on 7 after the commit. B: finished on 6 just
# before the sync, its answer read after the commit to 7.
@pytest.mark.parametrize("answered", ["7", "6"])
def test_a_sample_that_crossed_a_commit_is_accepted(answered):
    clients: list[AtomMeshRolloutClient] = []

    def commit_then_answer(body):
        # The sync to 7 commits while the request is in flight.
        clients[0].committed_version = 7
        return 200, _generate_body(body, weight_version=answered)

    stub = Stub(commit_then_answer)

    async def use(url):
        clients.append(AtomMeshRolloutClient(url, committed_version=6))
        return await clients[0].generate_batch([[77]], {"max_tokens": 8})

    rows = asyncio.run(stub.run(use))
    assert rows[0]["token_ids"] == OUTPUTS[77]["token_ids"]
    assert clients[0].sample_meta == [
        {
            "prompt_token_ids": [77],
            "request_id": "generate-3",
            "weight_version": int(answered),
            "cached_tokens": 3,
        }
    ]


def test_sample_meta_names_each_sample_prompt():
    # The stub's request id is generate-<output length>, distinct per prompt.
    prompts = [[151644, 872, 198, 785], [9906, 11], [77]]
    client, _ = _generate(Stub(), prompts, {"max_tokens": 8})

    by_id = {m["request_id"]: m["prompt_token_ids"] for m in client.sample_meta}
    assert by_id == {
        "generate-4": [151644, 872, 198, 785],
        "generate-1": [9906, 11],
        "generate-3": [77],
    }


@pytest.mark.parametrize(
    "answer, match",
    [
        (lambda body: (500, {"error": "boom"}), "500"),
        (
            lambda body: (
                200,
                {
                    **_generate_body(body),
                    "meta_info": {
                        **_generate_body(body)["meta_info"],
                        "finish_reason": {"type": "abort"},
                    },
                },
            ),
            "abort",
        ),
        (
            lambda body: (
                200,
                {
                    **_generate_body(body),
                    "meta_info": {
                        **_generate_body(body)["meta_info"],
                        "output_token_logprobs": [[-0.1, 902, None]],
                    },
                },
            ),
            "1 logprobs for 3 output ids",
        ),
    ],
)
def test_a_failed_request_raises(answer, match):
    with pytest.raises(AtomMeshRolloutError, match=match):
        _generate(Stub(answer), [[77]], {"max_tokens": 8, "logprobs": 0})
