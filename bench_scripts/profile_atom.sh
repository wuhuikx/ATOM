export VLLM_TORCH_PROFILER_DIR="qwen_atom"
export VLLM_TORCH_PROFILER_WITH_STACK=1
export VLLM_TORCH_PROFILER_RECORD_SHAPES=1
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1

model="/data/models/Qwen3-235B-A22B-Instruct-2507-FP8/"
vllm bench serve \
    --host localhost \
    --port 8000 \
    --model ${model} \
    --dataset-name random \
    --random-input-len 4096 \
    --random-output-len 10 \
    --max-concurrency 64 \
    --num-prompts 196 \
    --percentile-metrics ttft,tpot,itl,e2el \
    --ignore-eos \
    --profile \
    # --seed 123 \
    # --request-rate 2 \
    2>&1 | tee log.client.log
