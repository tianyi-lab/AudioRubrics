#!/bin/bash
# =============================================================================
# GRPO Baseline — audio encoder TRAINABLE (matches VPPO-RL / PAPO default).
#   1000 steps, same data + hyperparameters as before, but audio_tower now
#   backprops (previously hardcoded frozen). Vision tower remains frozen
#   (unused on AVQA).
# =============================================================================
set -e

# Encoder freeze control (handled in grpo_trainer.py via env vars).
export FREEZE_VISUAL=true
export FREEZE_AUDIO=false

export WORK_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$WORK_DIR/src"

export DEBUG_MODE="false"
export LOG_MODE="true"
export WANDB_MODE="offline"
export WANDB_START_METHOD="thread"
export SWANLAB_MODE="offline"
export PLOG="false"
export USE_VLLM=0
export USE_LOCAL_SAM=0

NVIDIA_BASE="/home/jovyan/envs/audio/lib/python3.12/site-packages/nvidia"
NVIDIA_LIBS=$(find "$NVIDIA_BASE" -name lib -type d | paste -sd:)
NVIDIA_INCS=$(find "$NVIDIA_BASE" -name include -type d | paste -sd:)
export CPATH="${NVIDIA_INCS}:${CPATH}"
export LIBRARY_PATH="${NVIDIA_LIBS}:${LIBRARY_PATH}"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH}"
export DS_SKIP_CUDA_CHECK=1
export CUDA_HOME="/home/jovyan/envs/audio"
export PATH="/home/jovyan/envs/audio/bin:$PATH"
export PYTORCH_ALLOC_CONF=expandable_segments:True

MODEL_PATH="${WORK_DIR}/models/Qwen/Qwen2.5-Omni-3B"
export LOG_PATH="train_logs/avqa_grpo_baseline"
mkdir -p "${WORK_DIR}/${LOG_PATH}"
export TRAIN_PATH="${WORK_DIR}/${LOG_PATH}"

ATTN_IMPL="flash_attention_2"

CUDA_VISIBLE_DEVICES=0 /home/jovyan/envs/audio/bin/torchrun \
    --nproc_per_node=1 --nnodes=1 --node_rank=0 \
    --master_addr="127.0.0.1" --master_port=12443 \
    -m omni_r1.grpo \
    --output_dir "$TRAIN_PATH" \
    --model_name_or_path "$MODEL_PATH" \
    --datasets_json 'datasets_full.json' \
    --training_datasets 'AVQA' \
    --deepspeed local_scripts/zero3_opt_offload.json \
    --max_prompt_length 4096 \
    --max_completion_length 500 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-6 \
    --lr_scheduler_type "cosine" \
    --bf16 \
    --logging_steps 1 \
    --gradient_checkpointing false \
    --len_control false \
    --attn_implementation "$ATTN_IMPL" \
    --max_steps 1000 \
    --run_name avqa_grpo_baseline \
    --save_steps 200 \
    --beta 0.001 \
    --alpha_k 0.0 \
    --alpha_a 1.0 \
    --alpha_g 0.0 \
    --max_grad_norm 1.0 \
    --save_only_model true \
    --report_to none \
    --num_generations 8 \
    --reward_funcs accuracy format \
    --use_sr1 false \
    --stage1_acc_weight 0.9 \
    --stage1_format_weight 0.1
