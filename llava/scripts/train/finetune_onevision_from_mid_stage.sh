export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

############################################
# Get the list of GPU indices
gpu_indices=$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)

# Check if GPU indices were found
if [ -n "$gpu_indices" ]; then
    export CUDA_VISIBLE_DEVICES=$gpu_indices
    echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"
else
    echo "No GPUs found or an error occurred."
fi
############################################

LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
MID_RUN_NAME=$1 
DATA_PATH=$2
LEARNING_RATE=$3
EPOCH=$4
MAX_SEQ_LENGTH=$5
MID_RUN_NAME="${MID_RUN_NAME}-seq_len_${MAX_SEQ_LENGTH}-lr_${LEARNING_RATE}-ep_${EPOCH}"
#$LLM_VERSION # this could also be the previous stage checkpoint
CKPT_PATH="lmms-lab/llava-onevision-qwen2-7b-mid-stage-a4"
IMAGE_FOLDER="/export/home/image"

if [ -z $ADDR ]; then
    echo "ADDR is empty"
    export ADDR=$(hostname -I | awk '{print $1}')
fi
if [ -z $PORT ]; then
    echo "PORT is empty"
    export PORT=12956
fi
if [ -z $NNODES ]; then
    echo "NNODES is empty"
    export NNODES=1
fi
if [ -z $RANK ]; then
    echo "RANK is empty"
    export RANK=0
fi

NGPU_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

NUM_GPUS=$((${NNODES} * ${NGPU_PER_NODE}))
# WORKERS=$((${NNODES} * ${NGPU_PER_NODE} * 4))

# if [ $WORKERS -gt 112 ]; then
#     WORKERS=112
# fi

global_batch_size=128
per_device_train_batch_size=1
gradient_accumulation_steps=$(($global_batch_size / ($per_device_train_batch_size * $NUM_GPUS)))
ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    llava/train/train_mem.py \
    --deepspeed llava/scripts/zero3.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --image_folder $IMAGE_FOLDER \
    --data_path $DATA_PATH \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter="llava/scripts/train/mm_projectors/7b/mm_projector.bin" \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "../checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs $EPOCH \
    --per_device_train_batch_size $per_device_train_batch_size \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length $MAX_SEQ_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32

# You can delete the sdpa attn_implementation if you want to use flash attn
