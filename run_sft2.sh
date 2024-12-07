#!/bin/bash

deepspeed \
    --num_gpus 8 \
    --num_nodes 2 \
    --hostfile hostfile \
    --master_addr 10.252.32.12 src/train.py \
    --deepspeed examples/deepspeed/ds_z2_config.json \
    --model_name_or_path /nas_data/userdata/randy/models/Llama-3.1-8B-Instruct \
    --stage sft \
    --do_train True \
    --finetuning_type lora \
    --dataset semi-fineweb-alpaca-en-zh-shuffling,cxmt-cptest-sft \
    --template llama3 \
    --cutoff_len 2048 \
    --max_samples 10000 \
    --overwrite_cache True \
    --num_train_epochs 4.0 \
    --preprocessing_num_workers 64 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.1 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --flash_attn fa2 \
    --ddp_timeout 180000000 \
    --neftune_noise_alpha 5 \
    --output_dir /nas_data/userdata/randy/models/cxmt/sft/lora/llama3.1-8b-instruct-sft-v1 \
    --overwrite_output_dir True \
    --logging_steps 10 \
    --save_steps 100 \
    --per_device_eval_batch_size 1 \
    --eval_strategy steps \
    --eval_steps 100 \
    --val_size 0.1 \
    --plot_loss \
    --bf16
