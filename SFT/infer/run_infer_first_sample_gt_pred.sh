#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/lyz}
CKPT=${CKPT:-/root/autodl-tmp/lyz/output/qwen3_vl_4b_vg_sft_full/sft_v2-20260626-014804/checkpoint-3514}

if [ ! -d "${CKPT}" ]; then
  echo "CKPT directory not found: ${CKPT}" >&2
  exit 1
fi

if ! compgen -G "${CKPT}/*.safetensors" >/dev/null && ! compgen -G "${CKPT}/pytorch_model*.bin" >/dev/null && ! compgen -G "${CKPT}/model*.bin" >/dev/null; then
  echo "No HuggingFace weight file found under CKPT: ${CKPT}" >&2
  echo "If this is a verl FSDP checkpoint, merge it first, for example:" >&2
  echo "  cd ${ROOT}/verl && /root/autodl-tmp/conda_envs/verl/bin/python -m verl.model_merger merge --backend fsdp --local_dir ${ROOT}/output/verl_grpo_vg_relation/qwen3_vl_4b_vg_relation_grpo_20260615_194423/global_step_240/actor --target_dir ${ROOT}/output/verl_grpo_vg_relation/qwen3_vl_4b_vg_relation_grpo_20260615_194423/global_step_640/actor_merged_hf --trust-remote-code" >&2
  exit 1
fi
TEST_IMAGES_JSON=${TEST_IMAGES_JSON:-${ROOT}/jsondata/vg_test_14700_image_paths.json}
TRAIN_DATASET_JSONL=${TRAIN_DATASET_JSONL:-/root/autodl-tmp/lyz/jsondata/erejin_datasets/origin/bbox0-1000/merged_5cls_0-1000_swift_structured.jsonl}
OUT_DIR=${OUT_DIR:-${ROOT}/output/erejin_sft}
SPLIT_TAG=${SPLIT_TAG:-test}

# 从CKPT路径提取训练标识和checkpoint步数，构建唯一输出名
# e.g. sftv2-20260609-165138/checkpoint-930 -> sftv2-20260609-165138_ckpt930
CKPT_PARENT=$(basename "$(dirname "${CKPT}")")
CKPT_STEP=$(basename "${CKPT}" | sed 's/checkpoint-/ckpt/')
INFER_TAG="${CKPT_PARENT}_${CKPT_STEP}_v2"

# Usage:
#   bash scripts/infer/run_infer_first_sample_gt_pred.sh          # infer first 500 test images
#   bash scripts/infer/run_infer_first_sample_gt_pred.sh 100      # infer first 100 test images
#   NUM_IMAGES=100 bash scripts/infer/run_infer_first_sample_gt_pred.sh
NUM_IMAGES=${1:-${NUM_IMAGES:-500}}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-6000}
MODEL_TYPE=${MODEL_TYPE:-qwen3_vl}

# TODO: TEMPERATURE需要设置为1 
TEMPERATURE=${TEMPERATURE:-1}  
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:-0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}  # 单卡评测默认只暴露 1 张卡
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}  # 双卡设置为2
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.95}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-32768}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-256}
VLLM_ENABLE_PREFIX_CACHING=${VLLM_ENABLE_PREFIX_CACHING:-true}
VLLM_LIMIT_MM_PER_PROMPT=${VLLM_LIMIT_MM_PER_PROMPT:-'{"image": 1}'}
VLLM_MM_PROCESSOR_CACHE_GB=${VLLM_MM_PROCESSOR_CACHE_GB:-4}
WRITE_BATCH_SIZE=${WRITE_BATCH_SIZE:-128}

# Lower image token cost is usually the largest speed win for VL batch inference.
MAX_PIXELS=${MAX_PIXELS:-1003520}
MIN_PIXELS=${MIN_PIXELS:-3136}

PYTHON=${PYTHON:-/root/autodl-tmp/conda_envs/ms-swift/bin/python}
SWIFT_BIN=${SWIFT_BIN:-/root/autodl-tmp/conda_envs/ms-swift/bin/swift}

export CUDA_VISIBLE_DEVICES
export MAX_PIXELS
export MIN_PIXELS
export PYTHONPATH="${ROOT}/ms-swift:${PYTHONPATH:-}"

mkdir -p "${OUT_DIR}"

INFER_DATA="${OUT_DIR}/vg_${SPLIT_TAG}_first_${NUM_IMAGES}.jsonl"
RESULT_PATH="${RESULT_PATH:-${OUT_DIR}/${INFER_TAG}_${SPLIT_TAG}${NUM_IMAGES}_pred.jsonl}"

"${PYTHON}" - "${TEST_IMAGES_JSON}" "${TRAIN_DATASET_JSONL}" "${INFER_DATA}" "${NUM_IMAGES}" <<'PYGEN'
import json
import sys
from pathlib import Path

test_images_json, dataset_jsonl, out_path, num_images_arg = sys.argv[1:]
num_images = int(num_images_arg)
if num_images <= 0:
    raise SystemExit(f"NUM_IMAGES must be positive, got {num_images}")

test_images_path = Path(test_images_json)
if not test_images_path.exists():
    raise SystemExit(f"test images json not found: {test_images_path}")

dataset_path = Path(dataset_jsonl)
if not dataset_path.exists():
    raise SystemExit(f"dataset jsonl not found: {dataset_path}")

with dataset_path.open("r", encoding="utf-8") as src:
    first_row = None
    for line in src:
        if line.strip():
            first_row = json.loads(line)
            break
if not first_row or "messages" not in first_row:
    raise SystemExit(f"no usable prompt row found in: {dataset_path}")
messages = first_row["messages"]
if not messages or messages[0].get("role") != "system":
    raise SystemExit("training dataset first row does not contain a system prompt")
if len(messages) < 2 or messages[1].get("role") != "user":
    raise SystemExit("training dataset first row does not contain a user prompt")
prompt_messages = [
    {"role": "system", "content": messages[0]["content"]},
    {"role": "user", "content": messages[1]["content"]},
]

out = Path(out_path)
out.parent.mkdir(parents=True, exist_ok=True)
count = 0
with out.open("w", encoding="utf-8") as f:
    with test_images_path.open("r", encoding="utf-8") as src:
        image_payload = json.load(src)
    image_paths = image_payload["image_paths"]
    for image_path in image_paths[:num_images]:
        image_id = Path(image_path).stem
        row = {
            "messages": prompt_messages,
            "images": [image_path],
            "image_id": image_id,
        }
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        count += 1

print(f"prepared_samples={count}")
print(f"infer_data={out}")
PYGEN

echo "ckpt=${CKPT}"
echo "test_images_json=${TEST_IMAGES_JSON}"
echo "train_dataset_jsonl=${TRAIN_DATASET_JSONL}"
echo "split_tag=${SPLIT_TAG}"
echo "infer_data=${INFER_DATA}"
echo "result_path=${RESULT_PATH}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "vllm_tp=${VLLM_TENSOR_PARALLEL_SIZE}"
echo "model_type=${MODEL_TYPE}"
echo "write_batch_size=${WRITE_BATCH_SIZE}"

${SWIFT_BIN} infer \
  --model "${CKPT}" \
  --model_type "${MODEL_TYPE}" \
  --infer_backend vllm \
  --val_dataset "${INFER_DATA}" \
  --result_path "${RESULT_PATH}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --top_k "${TOP_K}" \
  --repetition_penalty "${REPETITION_PENALTY}" \
  --stream false \
  --torch_dtype bfloat16 \
  --enable_thinking false \
  --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}" \
  --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
  --vllm_max_model_len "${VLLM_MAX_MODEL_LEN}" \
  --vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}" \
  --vllm_enable_prefix_caching "${VLLM_ENABLE_PREFIX_CACHING}" \
  --vllm_limit_mm_per_prompt "${VLLM_LIMIT_MM_PER_PROMPT}" \
  --vllm_mm_processor_cache_gb "${VLLM_MM_PROCESSOR_CACHE_GB}" \
  --write_batch_size "${WRITE_BATCH_SIZE}"

# The training jsonl prompt template is reused for test-image inference. ms-swift may emit a `labels` field for
# causal_lm datasets, so rewrite the result jsonl in place as pred-only.
"${PYTHON}" - "${RESULT_PATH}" <<'PYCLEAN'
import json
import sys
import tempfile
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"result file not found: {path}")
keep_keys = ["image_id", "images", "response", "messages", "logprobs"]
with path.open("r", encoding="utf-8") as src, tempfile.NamedTemporaryFile(
    "w", encoding="utf-8", dir=str(path.parent), delete=False
) as tmp:
    tmp_path = Path(tmp.name)
    for line in src:
        if not line.strip():
            continue
        row = json.loads(line)
        row.pop("labels", None)
        clean = {key: row[key] for key in keep_keys if key in row}
        tmp.write(json.dumps(clean, ensure_ascii=False) + "\n")
tmp_path.replace(path)
print(f"pred_only_result_path={path}")
PYCLEAN
