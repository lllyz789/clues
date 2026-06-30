#!/usr/bin/env bash
set -xeuo pipefail

# =============================================================================
# 8x80GB GPU configuration for 3B student + 3B teacher OPD training
# Layout: 6 GPUs actor/rollout + 2 GPUs teacher (2 replicas)
# =============================================================================

ROOT=${ROOT:-/root/autodl-tmp/lyz}
VERL_DIR=${VERL_DIR:-${ROOT}/verl}
CONDA_ENV_DIR=${CONDA_ENV_DIR:-/root/autodl-tmp/conda_envs/verl}
PYTHON_BIN=${PYTHON_BIN:-${CONDA_ENV_DIR}/bin/python}

MODEL_PATH=${MODEL_PATH:-${ROOT}/output/qwen3_vl_4b_vg_sft_full_mix/sft_mix-20260628-082902/checkpoint-626}
TEACHER_MODEL_PATH=${TEACHER_MODEL_PATH:-${ROOT}/output/teacher/model-20260628-172217/teacher_model}
VG_DIR=${VG_DIR:-${ROOT}/datasets/vg/VG_100K}
VG_DATA=${VG_DATA:-${ROOT}/datasets/vg/vg_augmentation_data.json}
DATA_DIR=${DATA_DIR:-${ROOT}/jsondata/trainRA_57723/verl_rl_56k_opd}
TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/train.parquet}
VAL_FILE=${VAL_FILE:-${DATA_DIR}/val.parquet}
REWARD_FILE=${REWARD_FILE:-${VERL_DIR}/verl/utils/reward_score/vg_relation_clue_opd.py}
PREDICATE_FILE=${PREDICATE_FILE:-${ROOT}/jsondata/erejin_datasets/vg_psg_predicate.txt}

PROJECT_NAME=${PROJECT_NAME:-verl_opd_vg_clue}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_vl_4b_vg_opd_8gpu_$(date +%Y%m%d_%H%M%S)}
CKPTS_DIR=${CKPTS_DIR:-${ROOT}/output/verl_opd_vg_clue/${EXPERIMENT_NAME}}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${CKPTS_DIR}/rollout_data}

# --- GPU layout ---
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-6}
NNODES=${NNODES:-1}
DISTILLATION_N_GPUS_PER_NODE=${DISTILLATION_N_GPUS_PER_NODE:-2}
DISTILLATION_NNODES=${DISTILLATION_NNODES:-1}
TEACHER_TP=${TEACHER_TP:-1}
TEACHER_GPU_MEM_UTIL=${TEACHER_GPU_MEM_UTIL:-0.85}

# --- Batch sizes (aggressive for 6x80GB + 3B model) ---
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-4}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-5500}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}

# --- Optimizer ---
ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.02}
USE_KL_LOSS=${USE_KL_LOSS:-False}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

# --- Rollout (no TP needed for 3B, maximize throughput) ---
ROLLOUT_TP=${ROLLOUT_TP:-1}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.8}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.9}
ROLLOUT_TOP_K=${ROLLOUT_TOP_K:-50}
ROLLOUT_BATCH_SIZE=$((TRAIN_BATCH_SIZE * ROLLOUT_N))
if [ -z "${AGENT_LOOP_NUM_WORKERS+x}" ]; then
    AGENT_LOOP_NUM_WORKERS=16
    while [ "${AGENT_LOOP_NUM_WORKERS}" -gt 1 ] && [ $((ROLLOUT_BATCH_SIZE % AGENT_LOOP_NUM_WORKERS)) -ne 0 ]; do
        AGENT_LOOP_NUM_WORKERS=$((AGENT_LOOP_NUM_WORKERS - 1))
    done
elif [ $((ROLLOUT_BATCH_SIZE % AGENT_LOOP_NUM_WORKERS)) -ne 0 ]; then
    echo "AGENT_LOOP_NUM_WORKERS=${AGENT_LOOP_NUM_WORKERS} must divide TRAIN_BATCH_SIZE*ROLLOUT_N=${ROLLOUT_BATCH_SIZE}" >&2
    exit 1
fi
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.85}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-131072}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-32}
ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-True}

# --- No offload needed (3B on 80GB has plenty of room) ---
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
ACTOR_OFFLOAD_POLICY=${ACTOR_OFFLOAD_POLICY:-False}

# --- Distillation ---
DISTILLATION_LOSS_MODE=${DISTILLATION_LOSS_MODE:-k1}
DISTILLATION_TOPK=${DISTILLATION_TOPK:-50}
DISTILLATION_LOSS_COEF=${DISTILLATION_LOSS_COEF:-1.0}
DISTILLATION_USE_TASK_REWARDS=${DISTILLATION_USE_TASK_REWARDS:-True}
DISTILLATION_USE_POLICY_GRADIENT=${DISTILLATION_USE_POLICY_GRADIENT:-True}

# --- Training schedule ---
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-100}
TEST_FREQ=${TEST_FREQ:-500}
VAL_SIZE=${VAL_SIZE:-128}
TRAINER_LOGGER=${TRAINER_LOGGER:-'["console","swanlab"]'}

export CUDA_VISIBLE_DEVICES
export PYTHONPATH=${VERL_DIR}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-}
export TEACHER_RELATION_EMBED_MODEL_PATH="${ROOT}/model/all-MiniLM-L6-v2"
export TEACHER_RELATION_EMBED_DEVICE=${TEACHER_RELATION_EMBED_DEVICE:-cpu}
export VG_RELATION_EMBED_MODEL_PATH="${ROOT}/model/all-MiniLM-L6-v2"
export VG_RELATION_EMBED_DEVICE=${VG_RELATION_EMBED_DEVICE:-cpu}

if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${VAL_FILE}" ]; then
    echo "Missing preconverted data files:" >&2
    echo "  ${TRAIN_FILE}" >&2
    echo "  ${VAL_FILE}" >&2
    echo "Run scripts/convert_vg_augmentation_to_verl_5cls.py first." >&2
    exit 1
fi
if [ ! -d "${TEACHER_MODEL_PATH}" ]; then
    echo "Missing teacher model checkpoint:" >&2
    echo "  ${TEACHER_MODEL_PATH}" >&2
    exit 1
fi

cd "${VERL_DIR}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo_sync \
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
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
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
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${ROLLOUT_TOP_K}" \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_LOOP_NUM_WORKERS}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    distillation.enabled=True \
    distillation.n_gpus_per_node="${DISTILLATION_N_GPUS_PER_NODE}" \
    distillation.nnodes="${DISTILLATION_NNODES}" \
    distillation.teacher_models.teacher_model.model_path="${TEACHER_MODEL_PATH}" \
    distillation.teacher_models.teacher_model.inference.name=vllm \
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size="${TEACHER_TP}" \
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization="${TEACHER_GPU_MEM_UTIL}" \
    distillation.teacher_models.teacher_model.inference.max_model_len="$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 1))" \
    distillation.teacher_models.teacher_model.inference.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    distillation.teacher_models.teacher_model.inference.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    distillation.teacher_models.teacher_model.inference.enable_chunked_prefill="${ROLLOUT_ENABLE_CHUNKED_PREFILL}" \
    distillation.teacher_models.teacher_model.inference.temperature=1.0 \
    +distillation.teacher_models.teacher_model.inference.engine_kwargs.vllm.max_logprobs="${DISTILLATION_TOPK}" \
    distillation.distillation_loss.loss_mode="${DISTILLATION_LOSS_MODE}" \
    distillation.distillation_loss.topk="${DISTILLATION_TOPK}" \
    distillation.distillation_loss.use_task_rewards="${DISTILLATION_USE_TASK_REWARDS}" \
    distillation.distillation_loss.distillation_loss_coef="${DISTILLATION_LOSS_COEF}" \
    distillation.distillation_loss.use_policy_gradient="${DISTILLATION_USE_POLICY_GRADIENT}" \
    trainer.balance_batch=True \
    trainer.logger="${TRAINER_LOGGER}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${NDEVICES_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.val_before_train=False \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "$@"
