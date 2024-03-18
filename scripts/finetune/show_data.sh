#!/usr/bin/env bash
set -x  # print the commands

export MASTER_PORT=12356  # You should set the same master_port in all the nodes

OUTPUT_DIR='/root/vit_g_hybrid_pt_1200e_door_ft'
DATA_PATH='/root/VideoMAEv2/dataset/ourData/'
MODEL_PATH='/root/vit_g_hybrid_pt_1200e_k710_ft.pth'
ROOT_PATH='/root/VideoMAEv2/dataset/ourData/'

N_NODES=1  # Number of nodes
GPUS_PER_NODE=1  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:3}  # Other training args

# Please refer to `run_class_finetuning.py` for the meaning of the following hyperreferences
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=0 --master_addr=localhost \
        show_data.py \
        --model vit_base_patch16_224 \
        --data_set ourData \
        --nb_classes 2 \
        --data_root ${ROOT_PATH}\
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --batch_size 1 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 10 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --num_workers 10 \
        --opt adamw \
        --lr 1e-3 \
        --drop_path 0.3 \
        --clip_grad 5.0 \
        --layer_decay 0.9 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.1 \
        --warmup_epochs 5 \
        --epochs 60 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --dist_eval --enable_deepspeed \
        ${PY_ARGS}