---
repo: wuhuikx/ATOM
integration_branch: atomesh-rl
never_touch: ROCm/ATOM, junyyang-amd/ATOM
design_entry: CLAUDE.md
gate_task: black --check . && bash tools/rl/unit_tests.sh && python -m pytest atomrl/tests/unit atomrl/tests/atomrl -m "not gpu and not multigpu" && (cd atom/mesh && cargo test -- --test-threads=1)
new_tests_dir: tests
publish_language: English
max_tasks: 2
---

# ATOMesh + ATOM + AtomRL on the atomesh-rl line

The work tracked here is wuhuikx/LLM-Booster#718 (under #456): Qwen3-8B BF16
DAPO through ATOMesh + ATOM + AtomRL, compared against AtomRL driving ATOM
directly. M1 is 4 training + 4 rollout GPUs on one 8xMI355X node, 4 TP1 ATOM
workers behind ATOMesh, weights over RCCL. M2 (8-GPU colocated, CUDA IPC) is
planned after M1.

- The line starts at junyyang-amd/ATOM `RL` @ 8064deeb, which carries the
  ATOMesh RL control plane (`/v1/rl/*`, `--enable-rl`) and the engine's RL
  routes. Its design is `docs/atomesh_rl_control_plane.md`.
- AtomRL comes from ROCm/ATOM#3355 @ 5ae8ecaa, imported under `atomrl/`.
- ROCm/ATOM and junyyang-amd/ATOM are never written to: no push, comment or
  PR. Agents never post to wuhuikx/LLM-Booster; the owner does.
- Engine and contract changes (the engine's RL routes, the AtomRL/ATOMesh
  contract) are reviewed by @junyyang-amd as well as the reviewer agent; a PR
  touching them names @junyyang-amd in its body.
- GPU runs (ATOM workers, ATOMesh, AtomRL training) need an exclusive
  8xMI355X the owner names. Without one, escalate; never use a node whose GPUs
  hold another user's job, and never kill another user's container.
- The gate needs no GPU. Design documents go in `docs/` and are listed from
  `CLAUDE.md`.
