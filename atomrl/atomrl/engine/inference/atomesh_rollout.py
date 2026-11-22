"""Rollout generation through ATOMesh instead of in-process ATOM actors.

``AtomMeshRolloutClient`` posts SGLang's ``/generate`` to an ATOMesh router
that fronts ATOM workers started with ``--enable-rl``. Each sample is its own
request (ATOM's ``/generate`` has no ``n``), so ATOMesh's ``prefix_hash``
policy routes it by its token ids; the rows come back in send order and in
the shape ``ATOMRayServer.generate_batch`` returns, so the trainer reads both
paths alike.

Every response names the weight version that produced it. The client accepts
a sample only when that version lies between the version it had committed
when it sent the request and the one committed when it read the answer; a
sample outside that range came from weights this client never committed, and
the call raises. Nothing is retried: a failed request fails the call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)

# AtomRL's rollout and validation sampling dicts carry only these keys
# (``RLTrainer._ray_sampling_params``, ``_ray_eval_sampling_params``).
# ATOM's ``/generate`` ignores keys it does not know, so anything else is
# refused here rather than dropped there without a word.
_SAMPLING_KEYS = frozenset({"max_tokens", "temperature", "top_p", "top_k", "logprobs"})
_FINISHED = frozenset({"stop", "length"})
# ATOMesh's long-rollout request timeout: a 4096-token sample under a full
# batch can take far longer than an HTTP client's default.
GENERATE_TIMEOUT_S = 14400.0


class AtomMeshRolloutError(RuntimeError):
    """A ``/generate`` that failed: a non-200 answer or an unfinished sample."""


class WeightVersionMismatch(AtomMeshRolloutError):
    """A sample generated under a weight version the client did not commit."""

    def __init__(
        self, version: str | None, low: int, high: int, request_id: str | None
    ) -> None:
        super().__init__(
            f"/generate {request_id} answered with weight_version {version!r}, "
            f"outside the committed range [{low}, {high}]"
        )
        self.version = version
        self.low = low
        self.high = high
        self.request_id = request_id


class AtomMeshRolloutClient:
    """Token-in generation through ATOMesh's ``/generate``.

    ``committed_version`` is the weight version the client last committed:
    0 for a fresh run, whose workers start with ``--weight-version 0``, then
    whatever each weight sync commits.

    ``sample_meta`` collects, per accepted sample and in completion order,
    its prompt's token ids, the worker's request id, ``weight_version`` and
    ``cached_tokens``: the record that shows whether a prompt sent again
    after a weight sync hit a prefix cached under the old weights.
    """

    def __init__(
        self,
        base_url: str,
        max_model_len: int = 0,
        committed_version: int = 0,
        timeout_s: float = GENERATE_TIMEOUT_S,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_model_len = int(max_model_len)
        self.committed_version = int(committed_version)
        self.timeout_s = float(timeout_s)
        self.sample_meta: list[dict[str, Any]] = []

    def build_request(
        self, prompt_ids: list[int], sampling: dict[str, Any]
    ) -> dict[str, Any]:
        """The ``/generate`` body for one sample of ``prompt_ids``."""
        unknown = set(sampling) - _SAMPLING_KEYS
        if unknown:
            raise ValueError(
                f"sampling keys {sorted(unknown)} have no /generate field; "
                f"known keys: {sorted(_SAMPLING_KEYS)}"
            )
        max_tokens = int(sampling.get("max_tokens", 128))
        if self.max_model_len > 0:
            max_tokens = min(max_tokens, max(1, self.max_model_len - len(prompt_ids)))
        # AtomRL asks for the sampled token's logprob with vLLM's logprobs=0.
        logprobs = sampling.get("logprobs")
        if logprobs is not None and not isinstance(logprobs, bool):
            logprobs = int(logprobs) >= 0
        return {
            "input_ids": list(prompt_ids),
            "sampling_params": {
                "max_new_tokens": max(1, max_tokens),
                "temperature": float(sampling.get("temperature", 1.0)),
                "top_p": float(sampling.get("top_p", 1.0)),
                "top_k": int(sampling.get("top_k", -1)),
            },
            "return_logprob": bool(logprobs),
        }

    async def generate_batch(
        self,
        prompts_token_ids: list[list[int]],
        sampling: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """One ``/generate`` per entry of ``prompts_token_ids``, concurrently.

        As with ``ATOMRayServer.generate_batch``, a prompt that should be
        sampled n times appears n times; the rows come back in input order.
        """
        bodies = [self.build_request(list(p), sampling) for p in prompts_token_ids]
        async with self._session() as session:
            return list(
                await asyncio.gather(*(self._generate_one(session, b) for b in bodies))
            )

    async def generate_group(
        self,
        prompt: list[int],
        num_generations: int,
        sampling: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """``num_generations`` samples of one prompt, one request each."""
        return await self.generate_batch([list(prompt)] * num_generations, sampling)

    def _session(self) -> aiohttp.ClientSession:
        # No connection cap: a rollout batch's samples are all in flight at
        # once, as they are on the direct path, and ATOMesh does the queueing.
        return aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(limit=0),
            timeout=aiohttp.ClientTimeout(total=self.timeout_s),
        )

    async def _generate_one(
        self, session: aiohttp.ClientSession, body: dict[str, Any]
    ) -> dict[str, Any]:
        v_send = self.committed_version
        async with session.post(f"{self.base_url}/generate", json=body) as resp:
            if resp.status != 200:
                raise AtomMeshRolloutError(
                    f"/generate answered {resp.status}: {await resp.text()}"
                )
            out = await resp.json()
        v_now = self.committed_version
        meta = out.get("meta_info") or {}
        request_id = meta.get("id")

        finish = (meta.get("finish_reason") or {}).get("type")
        if finish not in _FINISHED:
            raise AtomMeshRolloutError(
                f"/generate {request_id} finished with {meta.get('finish_reason')!r}"
            )

        version = meta.get("weight_version")
        try:
            accepted = v_send <= int(version) <= v_now
        except (TypeError, ValueError):
            accepted = False
        if not accepted:
            raise WeightVersionMismatch(version, v_send, v_now, request_id)

        token_ids = list(out.get("output_ids") or [])
        logprobs = None
        if body["return_logprob"]:
            records = meta.get("output_token_logprobs")
            if records is None or len(records) != len(token_ids):
                raise AtomMeshRolloutError(
                    f"/generate {request_id} returned "
                    f"{'no' if records is None else len(records)} logprobs "
                    f"for {len(token_ids)} output ids"
                )
            logprobs = [float(r[0]) for r in records]

        self.sample_meta.append(
            {
                "prompt_token_ids": body["input_ids"],
                "request_id": request_id,
                "weight_version": int(version),
                "cached_tokens": int(meta.get("cached_tokens") or 0),
            }
        )
        return {
            "text": out.get("text", ""),
            "prompt_token_ids": body["input_ids"],
            "token_ids": token_ids,
            "logprobs": logprobs,
        }
