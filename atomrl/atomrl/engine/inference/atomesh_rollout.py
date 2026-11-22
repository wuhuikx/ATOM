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

Weights reach the workers over one RCCL group that the trainer's rank 0 and
every worker join once, at setup (``join_weight_group``); the client finds
the workers and their TP sizes through ATOMesh's ``/v1/rl/workers`` and
places worker i at rank ``1 + sum of the TP sizes listed before it``. Each
``sync_weights`` then only streams: it pauses the workers, arms their
receivers through ATOMesh, runs the trainer's existing RCCL sender, checks
that every worker answered and reports the new version, commits it, and
resumes the workers whatever happened. A failed sync raises; it is neither
retried nor repaired.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from collections.abc import AsyncIterator
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
# The client waits this much longer than ATOMesh's per-call control timeout,
# so a worker that times out comes back as ATOMesh's failed call, not as a
# client timeout that names no worker.
_CONTROL_SLACK_S = 30.0


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


class WeightSyncError(AtomMeshRolloutError):
    """A weight-group join, sync or leave that did not complete everywhere."""


class AtomMeshRolloutClient:
    """Token-in generation through ATOMesh's ``/generate``.

    ``committed_version`` is the weight version the client last committed:
    0 for a fresh run, whose workers start with ``--weight-version 0``, then
    whatever each weight sync commits.

    ``sample_meta`` collects, per accepted sample and in completion order,
    its prompt's token ids, the worker's request id, ``weight_version`` and
    ``cached_tokens``: the record that shows whether a prompt sent again
    after a weight sync hit a prefix cached under the old weights.

    ``api_key`` is ATOMesh's ``--rl-api-key``, presented on every ``/v1/rl``
    call; ``control_timeout_s`` is ATOMesh's ``--rl-control-timeout-secs``.
    ``selector`` picks the workers the control fan-outs reach.
    """

    def __init__(
        self,
        base_url: str,
        max_model_len: int = 0,
        committed_version: int = 0,
        timeout_s: float = GENERATE_TIMEOUT_S,
        api_key: str | None = None,
        control_timeout_s: float = 600.0,
        selector: str = "engine=atom",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_model_len = int(max_model_len)
        self.committed_version = int(committed_version)
        self.timeout_s = float(timeout_s)
        self.sample_meta: list[dict[str, Any]] = []
        self.api_key = api_key
        self.control_timeout_s = float(control_timeout_s)
        self.selector = selector
        # Set by join_weight_group: the trainer's worker group, the group's
        # name and the ids of the workers that joined it.
        self.trainer = None
        self.group_name: str | None = None
        self.worker_ids: list[str] = []

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

    # ── Weight sync over /v1/rl ───────────────────────────────────────────

    async def join_weight_group(
        self,
        trainer,
        *,
        interface: str,
        timeout_s: int = 600,
        group_name: str | None = None,
    ) -> dict[str, Any]:
        """Form the trainer + all-workers RCCL group, once, before rollout.

        ``trainer`` is the actor worker group (``RayWorkerGroup``) whose rank 0
        hosts the rendezvous and later broadcasts. The trainer's join and one
        ``init_weights_update_group`` per worker go out together: every rank
        must call in before any returns.
        """
        if self.group_name is not None:
            raise WeightSyncError(f"already joined weight group {self.group_name}")
        group_name = group_name or f"atomesh-rl-weights-{os.getpid()}"
        async with self._control_session() as session:
            status, listing = await self._control(session, "GET", "/v1/rl/workers")
            if status != 200:
                raise WeightSyncError(f"/v1/rl/workers answered {status}: {listing}")
            engine = self.selector.partition("=")[2]
            workers = [w for w in listing["workers"] if w["engine"] == engine]
            if not workers:
                raise WeightSyncError(f"ATOMesh lists no {engine} worker")
            offsets, rank = [], 1
            for w in workers:
                # A DP worker's engines would each need their own ranks, and
                # ATOM refuses a weight stream under DP anyway.
                if not w.get("tp_size") or (w.get("dp_size") or 1) > 1:
                    raise WeightSyncError(
                        f"worker {w['id']} reports tp_size {w.get('tp_size')!r}, "
                        f"dp_size {w.get('dp_size')!r}; need a TP size and no DP"
                    )
                offsets.append(rank)
                rank += int(w["tp_size"])
            world_size = rank

            rendezvous = await trainer.call_single_async(
                0, "get_rdma_rendezvous", interface
            )
            address, port = str(rendezvous["address"]), int(rendezvous["port"])

            async def join(worker: dict[str, Any], offset: int) -> None:
                path = f"/v1/rl/workers/{worker['id']}/engine/init_weights_update_group"
                body = {
                    "master_address": address,
                    "master_port": port,
                    "rank_offset": offset,
                    "world_size": world_size,
                    "group_name": group_name,
                    "backend": "nccl",
                }
                status, out = await self._control(session, "POST", path, body)
                if status != 200:
                    raise WeightSyncError(
                        f"worker {worker['id']} did not join {group_name}: "
                        f"{status} {out}"
                    )

            await asyncio.gather(
                trainer.call_single_async(
                    0,
                    "init_rdma_weight_group",
                    address,
                    port,
                    world_size,
                    group_name,
                    int(timeout_s),
                ),
                *(join(w, off) for w, off in zip(workers, offsets)),
            )
        self.trainer = trainer
        self.group_name = group_name
        self.worker_ids = [w["id"] for w in workers]
        logger.info(
            "ATOMesh weight group %s ready: master=%s:%d world=%d workers=%s",
            group_name,
            address,
            port,
            world_size,
            self.worker_ids,
        )
        return {"master_addr": address, "master_port": port, "world_size": world_size}

    async def sync_weights(self, version: int, *, bucket_size_mb: int) -> dict:
        """Stream the trainer's weights to every worker as ``version``.

        Pauses the workers, arms their receivers through ATOMesh, runs the
        trainer's RCCL sender, confirms every worker joined at setup answered
        and now reports ``str(version)``, commits ``version`` and resumes the
        workers. Returns the sender's statistics. Raises ``WeightSyncError``
        on any failure, after resuming the workers.
        """
        if self.group_name is None:
            raise WeightSyncError("join_weight_group() has not run")
        version = int(version)
        async with self.paused() as session:
            # Arm the receivers before the sender starts and await both
            # together: the broadcast needs every rank inside it at once.
            receive = asyncio.ensure_future(
                self._fanout(
                    session,
                    "POST",
                    "receive_weights_rdma",
                    {"group_name": self.group_name, "version": version},
                )
            )
            sends = self.trainer.execute_all_async(
                "send_weights_rdma", version=version, bucket_size_mb=int(bucket_size_mb)
            )
            received, *sent = await asyncio.gather(
                receive, *sends, return_exceptions=True
            )
            errors = [r for r in (received, *sent) if isinstance(r, BaseException)]
            if errors:
                raise WeightSyncError(
                    f"weight sync to version {version} failed: {errors[0]}"
                ) from errors[0]
            self._check_fanout("receive_weights_rdma", received)
            stats = next((s for s in sent if s.get("writer")), None)
            if stats is None:
                raise WeightSyncError("the trainer's sender returned no statistics")

            info = await self._fanout(session, "GET", "model_info")
            reported = {
                wid: (out.get("body") or {}).get("weight_version")
                for wid, out in info.get("results", {}).items()
            }
            missed = {
                wid: reported.get(wid)
                for wid in self.worker_ids
                if reported.get(wid) != str(version)
            }
            if missed:
                raise WeightSyncError(
                    f"workers not at weight_version {str(version)!r} after the "
                    f"sync: {missed}"
                )
            self.committed_version = version
        logger.info("ATOMesh weight sync committed version %d: %s", version, stats)
        return stats

    @contextlib.asynccontextmanager
    async def paused(self) -> AsyncIterator[aiohttp.ClientSession]:
        """Pause every selected worker; resume them on every exit.

        Yields the control session. A failed pause still resumes, since some
        workers may have paused; a failed resume raises.
        """
        async with self._control_session() as session:
            try:
                info = await self._fanout(session, "POST", "pause_generation", {})
                self._check_fanout("pause_generation", info)
                yield session
            finally:
                info = await self._fanout(session, "POST", "continue_generation", {})
                self._check_fanout("continue_generation", info)

    async def leave_weight_group(self) -> None:
        """Have every worker and the trainer leave the group, at shutdown."""
        if self.group_name is None:
            return
        async with self._control_session() as session:
            info = await self._fanout(
                session,
                "POST",
                "destroy_weights_update_group",
                {"group_name": self.group_name},
            )
        await self.trainer.call_single_async(0, "destroy_rdma_weight_group")
        self.group_name = None
        self._check_fanout("destroy_weights_update_group", info)

    def _control_session(self) -> aiohttp.ClientSession:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        return aiohttp.ClientSession(
            headers=headers,
            timeout=aiohttp.ClientTimeout(
                total=self.control_timeout_s + _CONTROL_SLACK_S
            ),
        )

    async def _control(
        self,
        session: aiohttp.ClientSession,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        params: dict[str, str] | None = None,
    ) -> tuple[int, Any]:
        async with session.request(
            method, f"{self.base_url}{path}", json=body, params=params
        ) as resp:
            if resp.content_type == "application/json":
                return resp.status, await resp.json()
            return resp.status, await resp.text()

    async def _fanout(
        self,
        session: aiohttp.ClientSession,
        method: str,
        engine_path: str,
        body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One engine call to every selected worker; ATOMesh's report."""
        status, out = await self._control(
            session,
            method,
            f"/v1/rl/engine/{engine_path}",
            body,
            params={"selector": self.selector},
        )
        if status not in (200, 207) or not isinstance(out, dict):
            raise WeightSyncError(f"{engine_path} fan-out answered {status}: {out}")
        return out

    def _check_fanout(self, engine_path: str, report: dict[str, Any]) -> None:
        """Raise unless every selected worker answered with success and,
        once the group is joined, as many workers answered as joined it."""
        expected = len(self.worker_ids) or report.get("total")
        if report.get("failed") or report.get("total") != expected:
            raise WeightSyncError(
                f"{engine_path} reached {report.get('succeeded')} of "
                f"{len(self.worker_ids)} workers (ATOMesh matched "
                f"{report.get('total')}); failed: {report.get('failed')}"
            )
