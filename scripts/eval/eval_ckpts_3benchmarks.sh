#!/bin/bash
# =============================================================================
# Evaluate a list of GRPO checkpoints on MMAR + MMAU-test-mini + AIR-Bench
# Foundation via local vllm serve.
#
# Per checkpoint:
#   1) Merge thinker shards with original Qwen2.5-Omni-3B talker/code2wav
#      (scripts/merge_thinker_to_full.py) -> <ckpt>-full/
#   2) Launch vllm serve on $PORT
#   3) Run generate_answers_vllm.py for each dataset
#   4) Score with evaluation.py
#   5) Kill vllm and move on
#
# Idempotent: skips datasets whose preds.jsonl already has full coverage.
#
# Usage:
#   CKPT_BASE=/path/to/train_logs/<run_name> \
#   STEPS="200 400 600 800 1000" \
#   PORT=8002 \
#   OUT_BASE=/path/to/outputs \
#   bash scripts/eval/eval_ckpts_3benchmarks.sh
# =============================================================================
set +e

WORK=/home/jovyan/Omni-R1
PY_VLLM=/home/jovyan/envs/vllm/bin/python
PY_AUDIO=/home/jovyan/envs/audio/bin/python
MERGE=$WORK/scripts/merge_thinker_to_full.py
INFER=$WORK/scripts/eval/generate_answers_vllm.py
EVAL=$WORK/scripts/eval/evaluation.py
ORIG=$WORK/models/Qwen/Qwen2.5-Omni-3B
CKPT_BASE=${CKPT_BASE:-$WORK/train_logs/avqa_grpo_rar03_evolve_lmproj}
OUT_BASE=${OUT_BASE:-/home/jovyan/Audio_reason/outputs}
PORT=${PORT:-8002}
STEPS=${STEPS:-"200 400 600 800 1000"}
LOG=${LOG:-$WORK/train_logs/eval_ckpts.log}
TAG=${TAG:-$(basename $CKPT_BASE)}

# Register custom Qwen2.5OmniThinker config before vllm imports it.
REG="from transformers import AutoConfig; \
from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniThinkerConfig; \
AutoConfig.register('qwen2_5_omni_thinker', Qwen2_5OmniThinkerConfig)"

# Benchmark annotation / audio / sample count.
declare -A DS_ANN DS_AUD DS_N
DS_ANN[MMAR]=/home/jovyan/Audio_reason/datasets/MMAR/annotation/MMAR-meta.jsonl
DS_AUD[MMAR]=/home/jovyan/Audio_reason/datasets/MMAR/audio
DS_N[MMAR]=999
DS_ANN[MMAU]=/home/jovyan/Audio_reason/datasets/MMAU-test-mini/MMAU-test-mini-meta.jsonl
DS_AUD[MMAU]=/home/jovyan/Audio_reason/datasets/MMAU-test-mini/audio
DS_N[MMAU]=1000
DS_ANN[AIRBench]=/home/jovyan/Audio_reason/datasets/AIR-Bench-Foundation/annotation/AIR-Bench-Foundation.jsonl
DS_AUD[AIRBench]=/home/jovyan/Audio_reason/datasets/AIR-Bench-Foundation/audio
DS_N[AIRBench]=24683

log(){ echo "[$(date '+%H:%M:%S')] $*" | tee -a $LOG; }

for STEP in $STEPS; do
    CKPT=$CKPT_BASE/checkpoint-$STEP
    FULL=$CKPT_BASE/checkpoint-${STEP}-full
    [ ! -d "$CKPT" ] && { log "Step $STEP: ckpt missing, skip"; continue; }

    if [ ! -f "$FULL/model.safetensors.index.json" ]; then
        log "Step $STEP: merging thinker -> full"
        $PY_AUDIO $MERGE --ckpt_dir $CKPT --orig_dir $ORIG --out_dir $FULL >> $LOG 2>&1
    else
        log "Step $STEP: full ckpt exists"
    fi

    log "Step $STEP: launching vllm on port $PORT"
    $PY_VLLM -c "$REG; import vllm.entrypoints.cli.main; vllm.entrypoints.cli.main.main()" \
        serve $FULL --port $PORT --trust-remote-code --tensor-parallel-size 1 \
        --gpu-memory-utilization 0.85 --max-model-len 4096 \
        --served-model-name qwen2.5-omni-3b > /tmp/vllm_${TAG}_step${STEP}.log 2>&1 &
    PID=$!
    until curl -s http://localhost:$PORT/v1/models > /dev/null 2>&1; do
        sleep 3
        if ! kill -0 $PID 2>/dev/null; then log "Step $STEP: vllm crashed"; break; fi
    done

    for DS in MMAR MMAU AIRBench; do
        OUT=$OUT_BASE/${TAG}_${DS,,}_step${STEP}/${DS}
        mkdir -p $OUT
        if [ -f $OUT/preds.jsonl ] && [ $(wc -l < $OUT/preds.jsonl) -ge ${DS_N[$DS]} ]; then
            log "Step $STEP $DS: already done, skip"; continue
        fi
        log "Step $STEP $DS: infer ${DS_N[$DS]} samples"
        $PY_VLLM $INFER --input ${DS_ANN[$DS]} --audio_dir ${DS_AUD[$DS]} \
            --output $OUT/preds.jsonl --model qwen2.5-omni-3b \
            --base_url http://localhost:$PORT/v1 \
            --max_workers 8 --max_new_tokens 512 \
            >> $OUT/driver.log 2>&1 || true
        log "Step $STEP $DS: scoring"
        $PY_AUDIO $EVAL --input $OUT/preds.jsonl 2>&1 | grep -E "Total Accuracy" | tee -a $LOG
    done

    kill -9 $PID 2>/dev/null
    pkill -9 -f EngineCore 2>/dev/null
    sleep 5
done

log "=== ALL DONE ==="
