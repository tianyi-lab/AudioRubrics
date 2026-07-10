#!/bin/bash
# RAR evolving-rubric 容器适配版: 7B + lmproj + evolving rubric (gemini-3.1-pro judge)
set -e
export PYTHONUTF8=1
export LC_ALL=C.UTF-8
export FREEZE_VISUAL=true
export FREEZE_AUDIO=false
export FREEZE_AUDIO_BACKBONE_ONLY=true
export PYTORCH_ALLOC_CONF=expandable_segments:True

export WORK_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$WORK_DIR/src"
export DEBUG_MODE=false LOG_MODE=true WANDB_MODE=offline WANDB_START_METHOD=thread SWANLAB_MODE=offline PLOG=false
export USE_VLLM=0 USE_LOCAL_SAM=0

export CUDA_HOME=/usr/local/cuda
NVIDIA_BASE="/opt/venv/lib/python3.11/site-packages/nvidia"
if [ -d "$NVIDIA_BASE" ]; then
  export LD_LIBRARY_PATH="$(find "$NVIDIA_BASE" -name lib -type d | paste -sd:):${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi
export DS_SKIP_CUDA_CHECK=1
unset TORCH_CUDA_ARCH_LIST

# 运行时 modeling 补丁(纯音频修复)
PATCH_DST=$(python -c "import transformers.models.qwen2_5_omni.modeling_qwen2_5_omni as m; print(m.__file__)" 2>/dev/null)
[ -n "$PATCH_DST" ] && cp -f "$WORK_DIR/transformers/modeling_qwen2_5_omni.py" "$PATCH_DST" && echo "patched modeling -> $PATCH_DST"

: "${GEMINI_API_KEY:=${GEMINI_API_KEYS%%,*}}"; : "${GEMINI_API_KEY:?Set GEMINI_API_KEY or GEMINI_API_KEYS}"
MODEL_PATH="${MODEL_PATH:-/tmp/Qwen2.5-Omni-7B}"
NPROC="${NPROC:-4}"
MAX_STEPS="${MAX_STEPS:-1000}"
export LOG_PATH="${LOG_PATH:-train_logs/evolve_lmproj_7b}"
mkdir -p "${WORK_DIR}/${LOG_PATH}"
export TRAIN_PATH="${TRAIN_PATH:-${WORK_DIR}/${LOG_PATH}}"

torchrun --nproc_per_node=${NPROC} --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=${MASTER_PORT:-12468} \
    -m omni_r1.grpo \
    --output_dir "$TRAIN_PATH" \
    --model_name_or_path "$MODEL_PATH" \
    --datasets_json 'datasets_evolve.json' \
    --training_datasets 'AVQA' \
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 4096 \
    --max_completion_length 500 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "constant" \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --len_control false \
    --attn_implementation "flash_attention_2" \
    --max_steps ${MAX_STEPS} \
    --run_name ${RUN_NAME:-evolve_lmproj_7b} \
    --save_steps 200 \
    --beta 0.001 \
    --alpha_k 0.0 --alpha_a 1.0 --alpha_g 0.0 \
    --max_grad_norm 1.0 \
    --save_only_model true \
    --report_to none \
    --num_generations 8 \
    --reward_funcs accuracy format evolving_rubric overthinking \
    --use_sr1 false \
    --stage1_acc_weight 0.9 \
    --stage1_format_weight 0.1 \
    --rubric_path ${RUBRIC_PATH:-/rubrics/rubrics_avqa_train.jsonl} \
    --rubric_judge_model ${RUBRIC_JUDGE_MODEL:-gemini-3.1-pro-preview} \
    --rubric_judge_reasoning ${RUBRIC_JUDGE_REASONING:-low} \
    --rubric_weight ${RUBRIC_WEIGHT:-0.3} \
    --rubric_neutral 0.5 \
    --use_evolving_rubric true \
    --evolving_topk 5 \
    --evolving_max_new 3 \
    --evolving_system_prompt_path "${WORK_DIR}/data/evolving_rubric_system_prompt.md" \
    --evolving_uniform_weights false \
    --evolving_call_every 1 \
    --overthinking_weight ${OVERTHINK_WEIGHT:-0.2} \
    --overthinking_lmax ${OVERTHINK_LMAX:-256}
