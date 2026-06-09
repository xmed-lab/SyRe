#!/bin/bash
set -e

source /mnt/ali/fmodimg/miniconda3/etc/profile.d/conda.sh
conda activate syre118

export NCCL_SOCKET_IFNAME=bond0      # 明确告诉 NCCL 用 bond0（IPv4）
export NCCL_IB_GID_INDEX=0           # 强制使用 RoCE v1 / 禁止 IPv6 GID
export NCCL_DEBUG=INFO               # 先开调试，确认生效后可关掉
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
#set HF_ENDPOINT=https://hf-mirror.com
#export HF_ENDPOINT=https://hf-mirror.com
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600   # 例如 1 小时

echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
echo "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}"
echo "NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
echo "NCCL_DEBUG=${NCCL_DEBUG}"

# export PYTHONUNBUFFERED=1

#export CUDA_VISIBLE_DEVICES=0,1,2,3
GPUS_PER_NODE=8
# export HOSTFILE="./hostfile"


echo "[INFO] NNODES=${WORLD_SIZE} NODE_RANK=${RANK} MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT}"
echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} (GPUS_PER_NODE=${GPUS_PER_NODE})"

# Path to the checkpoint and output directory
export CKPT_PATH="/mnt/ali/fmodimg/SyRe_workdir/v16_final/model_bin/"
export OUTPUT_DIR_PATH="/mnt/ali/fmodimg/SyRe_workdir/SyRe"


mode=2d_train
mode_val=2d_test
text_prompts_path=/mnt/ali/fmodimg/Datasets2D/internvl_des_2d.json

# =========================
# DeepSpeed multi-node launch
# =========================

torchrun \
  --nnodes=${WORLD_SIZE} \
  --nproc_per_node=${GPUS_PER_NODE} \
  --node_rank=${RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train.py \
  --version $CKPT_PATH \
  --dataset_dir /mnt/ali/fmodimg/Datasets2D/ \
  --vision_pretrained /mnt/ali/fmodimg/SyRe/checkpoints/sam_vit_h_4b8939.pth \
  --exp_name $OUTPUT_DIR_PATH \
  --lora_r 16 \
  --lr 1e-4 \
  --pretrained \
  --epochs 10 \
  --batch_size 8 \
  --mask_validation \
  --mode $mode \
  --mode_val $mode_val \
  --text_prompts_path $text_prompts_path \
  --num_classes_per_sample 8 \
  --resume /mnt/ali/fmodimg/SyRe_workdir/SyRe/ckpt_model_last_epoch \
  --resume_from_mid \
  2>&1 | tee "./logs_new/train_syre_node${RANK}--contd.log"
    #--resume /mnt/ali/fmodimg/SyRe_workdir/v16_final/ckpt_model_last_epoch \
  #--resume_from_mid \

  
