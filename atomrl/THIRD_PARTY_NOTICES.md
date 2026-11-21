# AtomRL third-party notices

AtomRL and Lumen are distributed under the repository's MIT license. The
following incorporated or vendored sources retain their original Apache-2.0
terms and attribution. The complete Apache License 2.0 text is distributed as
`atom/mesh/LICENSE`.

## verl-derived sources

- Upstream: <https://github.com/volcengine/verl> (Apache-2.0)

- `atomrl/atomrl/algorithms/advantage_estimators.py`
- `atomrl/atomrl/algorithms/kl_controller.py`
- `atomrl/atomrl/algorithms/rollout_correction.py`
- `atomrl/atomrl/engine/training/base_engine.py`
- `atomrl/atomrl/engine/training/fsdp_engine.py`
- `atomrl/atomrl/data/sft_dataset.py`
- `atomrl/atomrl/rewards/math_reward.py`
- `atomrl/atomrl/engine/inference/bucketed_weight_transfer.py`
- `atomrl/atomrl/trainer/metric_utils.py`
- `atomrl/atomrl/utils/fsdp_utils.py`
- `atomrl/atomrl/utils/lr_scheduler.py`
- `atomrl/atomrl/utils/seqlen_balancing.py`
- `atomrl/atomrl/utils/torch_functional.py`
- `atomrl/atomrl/utils/ulysses.py`
- `atomrl/atomrl/utils/vocab_parallel.py`

These files preserve their upstream copyright and Apache-2.0 provenance
headers.

## vLLM-compatible weight transfer

- Upstream project: <https://github.com/vllm-project/vllm> (Apache-2.0)
- `atomrl/atomrl/engine/inference/bucketed_weight_transfer.py`

The AtomRL implementation is self-contained and documents its direct `verl`
source path above. This entry records the compatible vLLM weight-transfer
interface; the file is not relicensed as MIT.

## DeepSeek, MILES, and TorchSpec-derived sources

- DeepSeek-V4 source:
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/inference/kernel.py>
  (upstream terms apply):
  - `atomrl/atomrl/lumen/kernels/dsv4/quant/act_quant.py`
- MILES-derived kernels (Apache-2.0 provenance retained from the imported
  integration; the original MILES repository URL was not present in the
  imported files):
  - `atomrl/atomrl/lumen/kernels/dsv4/indexer/tilelang_*.py`
  - `atomrl/atomrl/lumen/kernels/dsv4/sparse_mla/tilelang_*.py`
- TorchSpec-derived sources (Apache-2.0 provenance retained; the upstream
  repository URL was not present in the imported files):
  - `atomrl/atomrl/trainer/bf16_optimizer.py`
  - `atomrl/atomrl/data/kimi_k25_parser.py`

The MILES and TorchSpec entries must retain their upstream notices and should
not be relicensed without the corresponding AMD OSS sign-off and source record.

## Vendored AITER and FlyDSL overlay

Files under `atomrl/lumen/examples/perf-veomni/aiter_overlay/` are vendored
from AITER/FlyDSL or maintained as an upstream-compatible overlay. Their
original SPDX identifiers and copyright notices are preserved; they are not
relicensed by the AtomRL/Lumen MIT migration.
