#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root, or set SYRE_ROOT explicitly.
SYRE_ROOT="${SYRE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${SYRE_ROOT}"

# User-configurable paths. Download the model and dataset described in README.md,
# and place the SAM checkpoint at checkpoints/sam_vit_h_4b8939.pth.
MODEL_PATH="${SYRE_MODEL:-./checkpoints/SyRe}"
DATASET_DIR="${SYRE_DATASET_DIR:-./data/SyReData}"
SAM_CHECKPOINT="${SYRE_SAM_CHECKPOINT:-./checkpoints/sam_vit_h_4b8939.pth}"
OUTPUT_DIR="${SYRE_OUTPUT_DIR:-./output/syre_2d}"
TEXT_PROMPTS_PATH="${SYRE_TEXT_PROMPTS_PATH:-${DATASET_DIR}/internvl_des_2d.json}"
RESUME_DIR="${SYRE_RESUME:-}"

MODE="${SYRE_MODE:-2d_train}"
MODE_VAL="${SYRE_MODE_VAL:-2d_test}"
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Dataset directory not found: ${DATASET_DIR}" >&2
  exit 1
fi
if [[ ! -f "${SAM_CHECKPOINT}" ]]; then
  echo "SAM checkpoint not found: ${SAM_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${TEXT_PROMPTS_PATH}" ]]; then
  echo "Text prompt file not found: ${TEXT_PROMPTS_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}" logs

TRAIN_ARGS=(
  --version "${MODEL_PATH}"
  --dataset_dir "${DATASET_DIR}"
  --vision_pretrained "${SAM_CHECKPOINT}"
  --exp_name "${OUTPUT_DIR}"
  --lora_r 16
  --lr 1e-4
  --pretrained
  --epochs 10
  --batch_size 8
  --grad_accumulation_steps 10
  --mask_validation
  --mode "${MODE}"
  --mode_val "${MODE_VAL}"
  --text_prompts_path "${TEXT_PROMPTS_PATH}"
  --num_classes_per_sample 8
)

if [[ -n "${RESUME_DIR}" ]]; then
  TRAIN_ARGS+=(--resume "${RESUME_DIR}")
fi

torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --node_rank="${NODE_RANK}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  train.py "${TRAIN_ARGS[@]}" \
  2>&1 | tee "logs/train_syre_node${NODE_RANK}.log"
