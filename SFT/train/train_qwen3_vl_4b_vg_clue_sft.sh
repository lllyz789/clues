#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/lyz
MODEL=${MODEL:-/root/autodl-tmp/lyz/output/qwen3_vl_4b_vg_sft_full/sft_v2-20260626-234146/checkpoint-1758}
DATASET=${DATASET:-/root/autodl-tmp/lyz/jsondata/erejin_datasets/clue_datas/vg_5cls_clue_sft.jsonl}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/output/qwen3_vl_4b_vg_sft_clue/clue_sft-${TIMESTAMP}}
LOG_FILE=${LOG_FILE:-${OUTPUT_DIR}/train.log}
ATTN_IMPL=${ATTN_IMPL:-flash_attn}
PADDING_FREE=${PADDING_FREE:-false}
NUM_EPOCHS=${NUM_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-5e-6}
MAX_LENGTH=${MAX_LENGTH:-8192}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:--1}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-2}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DEEPSPEED=${DEEPSPEED:-zero3}
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}

RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
AUTO_RESUME=${AUTO_RESUME:-false}
RESUME_ONLY_MODEL=${RESUME_ONLY_MODEL:-true}
CREATE_CHECKPOINT_SYMLINK=${CREATE_CHECKPOINT_SYMLINK:-true}

mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-2}
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ "${AUTO_RESUME}" == "true" && -z "${RESUME_FROM_CHECKPOINT}" ]]; then
    RESUME_FROM_CHECKPOINT=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" && ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "RESUME_FROM_CHECKPOINT not found: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 2
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
    RESUME_ARGS+=(--resume_only_model "${RESUME_ONLY_MODEL}")
fi

echo "=== Stage 2: Clue SFT Training ==="
echo "model=${MODEL}"
echo "dataset=${DATASET}"
echo "output_dir=${OUTPUT_DIR}"
echo "lr=${LR}"
echo "epochs=${NUM_EPOCHS}"
echo "batch_size=${BATCH_SIZE} x grad_accum=${GRAD_ACCUM}"
echo "max_length=${MAX_LENGTH}"
echo ""

set +e
swift sft \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --load_from_cache_file "${LOAD_FROM_CACHE_FILE}" \
    --split_dataset_ratio 0 \
    --tuner_type full \
    --torch_dtype bfloat16 \
    --attn_impl "${ATTN_IMPL}" \
    --padding_free "${PADDING_FREE}" \
    --freeze_llm false \
    --freeze_vit false \
    --freeze_aligner false \
    --num_train_epochs "${NUM_EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --learning_rate "${LR}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --save_strategy epoch \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --logging_steps 10 \
    --max_length "${MAX_LENGTH}" \
    --output_dir "${OUTPUT_DIR}" \
    --add_version false \
    --create_checkpoint_symlink "${CREATE_CHECKPOINT_SYMLINK}" \
    --lr_scheduler_type cosine \
    --weight_decay 0.01 \
    --max_grad_norm 1.0 \
    --warmup_ratio 0.05 \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --deepspeed "${DEEPSPEED}" \
    --save_only_model false \
    --report_to tensorboard \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e
exit "${STATUS}"
