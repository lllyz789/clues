#!/usr/bin/env bash
''':'
set -euo pipefail

ROOT=/root/autodl-tmp/lyz
MODEL=${MODEL:-/root/autodl-tmp/lyz/model/Qwen3-VL-4B-Instruct}
DATASET=${DATASET:-/root/autodl-tmp/lyz/jsondata/erejin_datasets/origin/bbox0-1000/vg_5cls_0-1000_swift_structured.jsonl}
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-${ROOT}/output/qwen3_vl_4b_vg_sft_full/sft_v2-${TIMESTAMP}}
LOG_FILE=${LOG_FILE:-${OUTPUT_DIR}/train.log}
ATTN_IMPL=${ATTN_IMPL:-flash_attn}
PADDING_FREE=${PADDING_FREE:-false}
NUM_EPOCHS=${NUM_EPOCHS:-2}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-1e-5}
MAX_LENGTH=${MAX_LENGTH:-5500}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:--1}
DATASET_NUM_PROC=${DATASET_NUM_PROC:-2}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DEEPSPEED=${DEEPSPEED:-zero3}
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-true}


# Resume switches:
#   RESUME_FROM_CHECKPOINT=/path/to/checkpoint-xxx  resume from a specific checkpoint.
#   AUTO_RESUME=true OUTPUT_DIR=/existing/run        resume from the latest checkpoint in OUTPUT_DIR.
#   RESUME_ONLY_MODEL=true                           load weights only, not optimizer/scheduler states.
RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT:-}
AUTO_RESUME=${AUTO_RESUME:-false}
RESUME_ONLY_MODEL=${RESUME_ONLY_MODEL:-false}
CREATE_CHECKPOINT_SYMLINK=${CREATE_CHECKPOINT_SYMLINK:-true}

mkdir -p "${OUTPUT_DIR}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export NPROC_PER_NODE=${NPROC_PER_NODE:-1}
# 每张图片最大视觉token数，控制图片分辨率/显存占用
export IMAGE_MAX_TOKEN_NUM=${IMAGE_MAX_TOKEN_NUM:-1024}
# PyTorch显存分配策略，expandable_segments减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ "${AUTO_RESUME}" == "true" && -z "${RESUME_FROM_CHECKPOINT}" ]]; then
    RESUME_FROM_CHECKPOINT=$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1 || true)
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" && ! -d "${RESUME_FROM_CHECKPOINT}" ]]; then
    echo "RESUME_FROM_CHECKPOINT not found: ${RESUME_FROM_CHECKPOINT}" >&2
    exit 2
fi

if [[ -n "${RESUME_FROM_CHECKPOINT}" && "${RESUME_ONLY_MODEL}" != "true" && "${SAVE_ONLY_MODEL}" == "true" ]]; then
    echo "Full checkpoint resume requires SAVE_TRAIN_STATE=true or SAVE_ONLY_MODEL=false." >&2
    echo "Current SAVE_ONLY_MODEL=true means only model weights are saved, without optimizer/scheduler state." >&2
    exit 2
fi

RESUME_ARGS=()
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
    RESUME_ARGS+=(--resume_from_checkpoint "${RESUME_FROM_CHECKPOINT}")
    RESUME_ARGS+=(--resume_only_model "${RESUME_ONLY_MODEL}")
fi

set +e
# ==================== swift sft 参数说明 ====================
# --model              基础模型路径或ModelScope模型ID
# --dataset            训练数据集路径，JSONL格式(messages+images)
# --load_from_cache_file  加载已缓存的预处理数据，加速二次启动
# --split_dataset_ratio   验证集划分比例，0表示不划分验证集
# --tuner_type         微调类型：full=全参数微调, lora=LoRA微调
# --torch_dtype        训练精度：bfloat16混合精度，节省显存且数值稳定
# --attn_impl          注意力实现：flash_attn(快+省显存), sdpa, eager
# --padding_free       无padding训练，多条样本拼接为一个序列以提高GPU利用率
# --freeze_llm         是否冻结LLM主干参数（false=训练LLM）
# --freeze_vit         是否冻结视觉编码器ViT（false=训练ViT）
# --freeze_aligner     是否冻结视觉-语言对齐层（false=训练aligner）
# --num_train_epochs   训练轮数
# --per_device_train_batch_size  每张GPU的batch size
# --learning_rate      初始学习率，全参微调通常1e-5~2e-5
# --gradient_accumulation_steps  梯度累积步数，等效batch=4*4*2GPU=32
# --gradient_checkpointing  梯度检查点：用计算换显存，减少~60%显存占用
# --vit_gradient_checkpointing  ViT也启用梯度检查点，进一步节省显存
# --save_strategy      保存策略：epoch=每个epoch结束保存一次checkpoint
# --save_total_limit   最多保留几个checkpoint，旧的自动删除
# --save_only_model    true只保存模型权重；false额外保存optimizer/scheduler/trainer/rng/DeepSpeed状态，可断点续训
# --resume_from_checkpoint  从指定checkpoint目录继续训练
# --resume_only_model  true只加载权重；false恢复完整训练状态
# --logging_steps      每N步记录一次loss等训练指标
# --max_length         单条样本最大token长度，超长会被截断
# --output_dir         输出目录：保存checkpoint和训练日志
# --add_version        不自动追加版本子目录(已手动加时间戳)
# --lr_scheduler_type  学习率调度：cosine退火，比linear收敛更平滑
# --weight_decay       权重衰减，L2正则化防止过拟合
# --max_grad_norm      梯度裁剪阈值，防止梯度爆炸
# --warmup_ratio       学习率预热比例，前5%步线性升至learning_rate
# --dataset_num_proc   数据预处理并行进程数
# --dataloader_num_workers  DataLoader加载数据的worker线程数
# --deepspeed          DeepSpeed ZeRO Stage3：参数+梯度+优化器状态全分片
# --create_checkpoint_symlink  创建last/best软链接，方便脚本定位最新checkpoint
# ===========================================================
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
    --warmup_ratio 0.03 \
    --dataset_num_proc "${DATASET_NUM_PROC}" \
    --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
    --deepspeed "${DEEPSPEED}" \
    --save_only_model false \
    "${RESUME_ARGS[@]}" \
    2>&1 | tee "${LOG_FILE}"
STATUS=${PIPESTATUS[0]}
set -e
exit "${STATUS}"
':'''
assert 0, "This is a Bash script. Run: bash train_qwen3_vl_4b_vg_full_sft.sh"



