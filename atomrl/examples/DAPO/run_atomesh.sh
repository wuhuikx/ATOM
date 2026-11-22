#!/usr/bin/env bash
# DAPO with the rollout behind ATOMesh, on one 8-GPU node: NUM_WORKERS TP1 ATOM
# workers (--enable-rl --weight-version 0) on ROLLOUT_GPUS, one ATOMesh router
# (--backend atom --enable-rl --policy prefix_hash, the long-rollout profile)
# in front of them, then the trainer through run_dapo.sh on the remaining GPUs
# with policy.generation.atom_cfg.transport=atomesh. The workers and ATOMesh
# stop when this script exits.
#
#   RL_ROOT=... DATA_ROOT=... ATOMESH_BIN=<target>/release/atomesh \
#     bash examples/DAPO/run_atomesh.sh
#
# The workers take their engine settings from the config's atom_cfg and
# vllm_cfg, mapped as ATOMReplicaManager's replicas take them on the direct
# path; RDMA_INTERFACE is the interface the trainer's rank 0 serves the RCCL
# rendezvous on (weight_sync.rdma.interface).
set -euo pipefail
: "${RL_ROOT:?}"; : "${DATA_ROOT:?}"; : "${ATOMESH_BIN:?path to the built atomesh binary}"
STEPS="${STEPS:-30}"
MODEL_PATH="${MODEL_PATH:-$DATA_ROOT/models/Qwen3-8B-Base}"
ATOMRL_DIR="${ATOMRL_DIR:-$RL_ROOT/ATOM/atomrl}"
ATOM_DIR="${ATOM_DIR:-$RL_ROOT/ATOM}"
AITER_DIR="${AITER_DIR:-$RL_ROOT/aiter}"
CONFIG="${CONFIG:-examples/DAPO/configs/dapo_qwen3_8b_atom_bf16_4p4_atomesh.yaml}"
ROLLOUT_GPUS=(${ROLLOUT_GPUS:-4 5 6 7})
WORKER_PORT="${WORKER_PORT:-18300}"  # worker i: HTTP WORKER_PORT+2i, engine +2i+1
MESH_PORT="${MESH_PORT:-18400}"
MESH_PROMETHEUS_PORT="${MESH_PROMETHEUS_PORT:-18401}"
RDMA_INTERFACE="${RDMA_INTERFACE:-ens11np0}"
RL_API_KEY="${RL_API_KEY:-$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-1200}"
LOG_DIR="${LOG_DIR:-$DATA_ROOT/logs/atomesh-$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$LOG_DIR"
cd "$ATOMRL_DIR"

MESH_URL="http://127.0.0.1:$MESH_PORT"
# The same overrides reach the worker settings below and the trainer.
OVERRIDES=(
  policy.model_name="$MODEL_PATH"
  policy.generation.atom_cfg.base_url="$MESH_URL"
  policy.generation.atom_cfg.api_key="$RL_API_KEY"
  weight_sync.rdma.interface="$RDMA_INTERFACE"
  # run_dapo.sh sets these two for in-process ATOM; the workers are not
  # colocated, so they stay resident and run the same compiled path.
  policy.generation.vllm_cfg.enforce_eager=false
  policy.generation.vllm_cfg.enable_sleep_mode=false
)

# Worker flags and ATOMesh's control timeout, read from the validated config.
SETTINGS_TEXT=$(PYTHONPATH="$ATOMRL_DIR" python3 - "$CONFIG" "${OVERRIDES[@]}" ${EXTRA_OVERRIDE:-} <<'PY'
import sys

from atomrl.core.config import AtomRLConfig

cfg = AtomRLConfig.from_yaml(sys.argv[1], overrides=sys.argv[2:])
atom = cfg.policy.generation.atom_cfg
vllm = cfg.policy.generation.vllm_cfg
if atom.transport != "atomesh":
    sys.exit(f"{sys.argv[1]}: atom_cfg.transport is {atom.transport!r}, not 'atomesh'")
if atom.tensor_parallel_size != 1 or atom.data_parallel_size != 1:
    sys.exit("run_atomesh.sh starts TP1 workers without DP")
args = [
    "--gpu-memory-utilization", str(atom.gpu_memory_utilization or vllm.gpu_memory_utilization),
    "--max-num-batched-tokens", str(vllm.max_num_batched_tokens),
    "--max-num-seqs", str(vllm.max_num_seqs),
    "--enable_chunked_prefill" if vllm.enable_chunked_prefill else "--no-enable_chunked_prefill",
    "--enable_prefix_caching" if atom.enable_prefix_caching else "--no-enable_prefix_caching",
]
if vllm.enforce_eager:
    args.append("--enforce-eager")
if vllm.trust_remote_code:
    args.append("--trust-remote-code")
if atom.kv_cache_dtype and atom.kv_cache_dtype != "auto":
    args += ["--kv_cache_dtype", atom.kv_cache_dtype]
if atom.max_model_len or vllm.max_model_len:
    args += ["--max-model-len", str(atom.max_model_len or vllm.max_model_len)]
if atom.engine_kwargs or atom.online_quant_config:
    sys.exit("atom_cfg.engine_kwargs and online_quant_config have no worker flag here")
print(cfg.weight_sync.timeout_s)
print(*args, sep="\n")
PY
)
mapfile -t SETTINGS <<<"$SETTINGS_TEXT"
CONTROL_TIMEOUT_S="${SETTINGS[0]}"
WORKER_ARGS=("${SETTINGS[@]:1}")

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill -- -"$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

# One process group each, so cleanup also stops a worker's engine processes.
# The ATOM environment is the one run_dapo.sh gives ATOM on the direct path.
URLS=()
for i in "${!ROLLOUT_GPUS[@]}"; do
  port=$((WORKER_PORT + 2 * i))
  HIP_VISIBLE_DEVICES="${ROLLOUT_GPUS[$i]}" AITER_LOG_LEVEL=WARNING \
    ATOM_FORCE_ATTN_TRITON="${ATOM_FORCE_ATTN_TRITON:-1}" \
    ATOMRL_ATOM_AITER_SRC="$AITER_DIR" NCCL_SOCKET_IFNAME="$RDMA_INTERFACE" \
    PYTHONPATH="$ATOMRL_DIR/examples/DAPO/atom_aiter_shim:$AITER_DIR:$ATOM_DIR:${PYTHONPATH:-}" \
    setsid python3 -m atom.entrypoints.openai_server --model "$MODEL_PATH" \
      --host 127.0.0.1 --server-port "$port" --port $((port + 1)) \
      --enable-rl --weight-version 0 "${WORKER_ARGS[@]}" \
      > "$LOG_DIR/atom-worker-$i.log" 2>&1 &
  PIDS+=($!)
  URLS+=("http://127.0.0.1:$port")
done

wait_for() {  # wait_for <what> <command...>: retry until it succeeds
  local what=$1 deadline=$((SECONDS + READY_TIMEOUT_S))
  shift
  until "$@" >/dev/null 2>&1; do
    for pid in "${PIDS[@]}"; do
      kill -0 "$pid" 2>/dev/null || { echo "a process died waiting for $what; logs in $LOG_DIR" >&2; exit 1; }
    done
    [ "$SECONDS" -lt "$deadline" ] || { echo "timed out waiting for $what" >&2; exit 1; }
    sleep 5
  done
}
for url in "${URLS[@]}"; do wait_for "$url" curl -sf "$url/health"; done

setsid "$ATOMESH_BIN" launch --host 127.0.0.1 --port "$MESH_PORT" \
  --prometheus-port "$MESH_PROMETHEUS_PORT" --backend atom --policy prefix_hash \
  --enable-rl --disable-health-check --disable-circuit-breaker \
  --request-timeout-secs 14400 --rl-api-key "$RL_API_KEY" \
  --rl-control-timeout-secs "$CONTROL_TIMEOUT_S" --worker-urls "${URLS[@]}" \
  > "$LOG_DIR/atomesh.log" 2>&1 &
PIDS+=($!)
mesh_lists_all() {
  curl -sf -H "Authorization: Bearer $RL_API_KEY" "$MESH_URL/v1/rl/workers" |
    python3 -c "import json, sys; sys.exit(json.load(sys.stdin)['total'] != $1)"
}
wait_for "ATOMesh to list ${#URLS[@]} workers" mesh_lists_all "${#URLS[@]}"
echo "ATOMesh at $MESH_URL over ${URLS[*]}; logs in $LOG_DIR"

# The trainer, on the GPUs the workers do not hold (cluster.gpus_per_node).
# KEEP_PROCS: run_dapo.sh's stale-process sweep would kill the workers' engines.
MODE=atombf16 STEPS="$STEPS" MODEL_PATH="$MODEL_PATH" CONFIG_OVERRIDE="$CONFIG" \
  KEEP_PROCS=1 EXTRA_OVERRIDE="${OVERRIDES[*]} ${EXTRA_OVERRIDE:-}" \
  bash "$ATOMRL_DIR/examples/DAPO/run_dapo.sh"
