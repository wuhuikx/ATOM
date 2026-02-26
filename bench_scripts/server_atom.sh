#!/bin/bash
export AITER_QUICK_REDUCE_QUANTIZATION=INT4
export ATOM_ENABLE_QK_NORM_ROPE_CACHE_QUANT_FUSION=1

export ATOM_TORCH_PROFILER_DIR="qwen_atom"
export ATOM_PROFILER_MORE=1

model_path=/data/models/Qwen3-235B-A22B-Instruct-2507-FP8/

python3 -m atom.entrypoints.openai_server --model ${model_path} \
	-tp 8 \
	--kv_cache_dtype fp8 \
	--enable-expert-parallel \
	--max-model-len 16384 \
	--max-num-batched-tokens 20000
