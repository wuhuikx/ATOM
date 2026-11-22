# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``/generate`` with ``return_logprob``: SGLang's
``meta_info.output_token_logprobs``, the same logprobs
``/v1/chat/completions`` returns for the same token ids and sampling, both
served by the real completion path over one fake engine."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.entrypoints.openai import api_server
from atom.model_engine.request import RequestOutput
from tests.entrypoints.test_api_compatibility import post
from tests.entrypoints.test_api_compatibility import server as _server_fixture
from tests.entrypoints.test_chat_logprobs import codec
from tests.logprob_helpers import token_logprob

server = _server_fixture
_generate_async = api_server.generate_async

PROMPT = [151644, 872, 198, 785]
# Token ids of the codec's vocabulary; 7 is its special end of sequence.
STEPS = [([0, 1], [-0.25, -1.5]), ([6], [-0.75]), ([7], [-0.125])]
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 20}


class _Engine:
    """The completion path's engine: preprocess streams the scripted
    outputs, each token with its engine record."""

    def __init__(self, steps):
        self.steps = steps
        self.prompts = []
        self.sampling = []
        self.requests = {}
        self.config = SimpleNamespace(
            model="/ckpt/base", max_model_len=8192, generation_config=None
        )
        self.io_processor = self
        self.core_mgr = self

    def preprocess(self, prompt, sampling_params, stream_callback=None, **_kwargs):
        self.prompts.append(list(prompt))
        self.sampling.append(sampling_params)
        seq = SimpleNamespace(
            id=7, num_prompt_tokens=len(prompt), max_tokens=sampling_params.max_tokens
        )
        self.requests[seq.id] = seq
        for i, (tokens, values) in enumerate(self.steps):
            last = i == len(self.steps) - 1
            stream_callback(
                RequestOutput(
                    request_id=seq.id,
                    output_tokens=tokens,
                    finished=last,
                    finish_reason="eos" if last else None,
                    logprobs=(
                        [token_logprob(v, 1, ((t, v),)) for t, v in zip(tokens, values)]
                        if sampling_params.logprobs
                        else None
                    ),
                    weight_version="7",
                )
            )
        return seq

    def add_request(self, _seqs):
        pass

    def abort_request(self, _seq_id):
        pass


@pytest.fixture
def rl(server, monkeypatch):
    """Chat's test server with the real ``generate_async`` over a fake
    engine, and the RL routes registered; the app's routes are put back
    afterwards."""
    routes = list(api_server.app.router.routes)
    engine_routes = set(api_server._ENGINE_ROUTES)
    monkeypatch.setattr(api_server, "generate_async", _generate_async)
    monkeypatch.setattr(api_server, "TokenBytes", lambda tokenizer: codec())
    monkeypatch.setattr(api_server, "record_nonstream_first_token", lambda: None)
    monkeypatch.setattr(
        api_server.tokenizer,
        "decode",
        lambda ids, **kw: f"decoded{list(ids)}",
    )

    def start(steps=STEPS):
        fake = _Engine(steps)
        monkeypatch.setattr(api_server, "engine", fake)
        api_server.register_rl_routes(SimpleNamespace(enable_rl=True))
        return fake

    yield start
    api_server.app.router.routes[:] = routes
    api_server._ENGINE_ROUTES.clear()
    api_server._ENGINE_ROUTES.update(engine_routes)
    api_server.app.openapi_schema = None


def _generate(**extra):
    return post(
        "/generate",
        {
            "input_ids": PROMPT,
            "sampling_params": {"max_new_tokens": 16, **SAMPLING},
            **extra,
        },
    )


def test_generate_logprobs_match_chat_completions_with_prompt_token_ids(rl):
    fake = rl()
    generated = _generate(return_logprob=True)
    assert generated.status_code == 200, generated.text
    chat = post(
        "/v1/chat/completions",
        {
            "messages": [{"role": "user", "content": "unused"}],
            "prompt_token_ids": PROMPT,
            "reasoning_effort": "none",
            "logprobs": True,
            "max_tokens": 16,
            **SAMPLING,
        },
    )
    assert chat.status_code == 200, chat.text

    # Both requests reached the engine alike.
    assert fake.prompts == [PROMPT, PROMPT]
    asked = [
        (p.max_tokens, p.temperature, p.top_p, p.top_k, bool(p.logprobs))
        for p in fake.sampling
    ]
    assert asked[0] == asked[1] == (16, 0.7, 0.9, 20, True)

    body = generated.json()
    ids = [t for tokens, _ in STEPS for t in tokens]
    assert body["output_ids"] == ids
    entries = body["meta_info"]["output_token_logprobs"]
    assert entries == [[v, t, None] for (ts, vs) in STEPS for t, v in zip(ts, vs)]
    # One per output token; chat leaves out the special end token's row.
    chat_rows = chat.json()["choices"][0]["logprobs"]["content"]
    assert [row["logprob"] for row in chat_rows] == [e[0] for e in entries[:-1]]
    # The text is decoded even though logprobs were asked for.
    assert body["text"] == f"decoded{ids}"
    assert body["meta_info"]["weight_version"] == "7"


def test_without_return_logprob_none_are_asked_or_answered(rl):
    fake = rl()
    reply = _generate()
    assert reply.status_code == 200, reply.text
    assert "output_token_logprobs" not in reply.json()["meta_info"]
    assert not fake.sampling[0].logprobs


def test_an_empty_output_answers_an_empty_list(rl):
    rl([([], [])])
    reply = _generate(return_logprob=True)
    assert reply.status_code == 200, reply.text
    assert reply.json()["meta_info"]["output_token_logprobs"] == []


def test_return_logprob_is_refused_where_the_engine_cannot_serve_it(rl, monkeypatch):
    fake = rl()
    monkeypatch.setattr(api_server, "synthetic_token_text", "x")
    reply = _generate(return_logprob=True)
    assert reply.status_code == 400
    assert "synthetic" in reply.json()["detail"]
    assert fake.prompts == []
