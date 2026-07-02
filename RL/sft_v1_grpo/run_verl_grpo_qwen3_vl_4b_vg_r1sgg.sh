#!/usr/bin/env bash
set -xeuo pipefail

ROOT=${ROOT:-/root/autodl-tmp/lyz}
VERL_DIR=${VERL_DIR:-${ROOT}/verl}
CONDA_ENV_DIR=${CONDA_ENV_DIR:-/root/autodl-tmp/conda_envs/verl}
PYTHON_BIN=${PYTHON_BIN:-${CONDA_ENV_DIR}/bin/python}

MODEL_PATH=${MODEL_PATH:-${ROOT}/output/qwen3_vl_4b_vg_sft_full/sft_v2-20260626-234146/checkpoint-1758}
VG_DIR=${VG_DIR:-${ROOT}/dataset/vg/vg_data/stanford_filtered}
VG_IMAGE_DIR=${VG_IMAGE_DIR:-${ROOT}/dataset/vg/VG_100K}
DATA_DIR=${DATA_DIR:-${ROOT}/jsondata/trainRA_57723/verl_rl_56k}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/val.parquet}
REWARD_FILE=${REWARD_FILE:-${VERL_DIR}/verl/utils/reward_score/vg_relation_no_clue.py}

PROJECT_NAME=${PROJECT_NAME:-verl_grpo_vg_relation}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_4b_vg_relation_grpo_$(date +%Y%m%d_%H%M%S)}
CKPTS_DIR=${CKPTS_DIR:-${ROOT}/output/verl_grpo_vg_relation/${EXPERIMENT_NAME}}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${CKPTS_DIR}/rollout_data}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-1}
NNODES=${NNODES:-1}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3072}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.02}
ENTROPY_COEFF=${ENTROPY_COEFF:-0.001}

ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-4}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
ROLLOUT_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
if [ -z "${AGENT_LOOP_NUM_WORKERS+x}" ]; then
    AGENT_LOOP_NUM_WORKERS=4
    while [ "${AGENT_LOOP_NUM_WORKERS}" -gt 1 ] && [ $((ROLLOUT_BATCH_SIZE % AGENT_LOOP_NUM_WORKERS)) -ne 0 ]; do
        AGENT_LOOP_NUM_WORKERS=$((AGENT_LOOP_NUM_WORKERS - 1))
    done
elif [ $((ROLLOUT_BATCH_SIZE % AGENT_LOOP_NUM_WORKERS)) -ne 0 ]; then
    echo "AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS} must divide TRAIN_BATCH_SIZE*ROLLOUT_N=${ROLLOUT_BATCH_SIZE}" >&2
    exit 1
fi
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.45}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-2}
ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}

ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-True}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-True}
ACTOR_OFFLOAD_POLICY=${ACTOR_OFFLOAD_POLICY:-True}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-600}
TEST_FREQ=${TEST_FREQ:-1000}

export CUDA_VISIBLE_DEVICES
export PYTHONPATH=${VERL_DIR}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}
SWANLAB_API_KEY=${SWANLAB_API_KEY:-OyVZPve2Ckn6i51nZdLHs}
if [ -z "${SWANLAB_API_KEY:-}" ]; then
    echo "Set SWANLAB_API_KEY before running training." >&2
    exit 1
fi
export SWANLAB_API_KEY
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-${ROOT}/output/swanlog}
export SWANLAB_MODE=${SWANLAB_MODE:-cloud}

if [ "${NDEVICES_PER_NODE}" -ne 1 ]; then
    echo "This script is tuned for a single visible GPU. Set NDEVICES_PER_NODE=1 or use a multi-GPU script." >&2
    exit 1
fi

if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${VAL_FILE}" ]; then
    echo "Missing verl parquet data:" >&2
    echo "  TRAIN_FILE=${TRAIN_FILE}" >&2
    echo "  VAL_FILE=${VAL_FILE}" >&2
    echo "Set DATA_DIR, TRAIN_FILE, or VAL_FILE to existing parquet files before running training." >&2
    exit 1
fi

cd "${VERL_DIR}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.image_key=images \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=False \
    data.truncation=right \
    reward.custom_reward_function.path="${REWARD_FILE}" \
    reward.custom_reward_function.name=compute_score \
    reward.reward_manager.name=naive \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.actor.clip_ratio_low=3e-4 \
    actor_rollout_ref.actor.clip_ratio_high=4e-4 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.offload_policy="${ACTOR_OFFLOAD_POLICY}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.enable_chunked_prefill="${ROLLOUT_ENABLE_CHUNKED_PREFILL}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.balance_batch=True \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NDEVICES_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.val_before_train=True \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "$@"
