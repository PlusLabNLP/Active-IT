#!/bin/bash
set -x
export TORCH_EXTENSIONS_DIR=/home/ponienkung # refer to https://github.com/pytorch/pytorch/issues/34238
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="$1"
export DataDir=$2
export run_name=$3
export additional_args=$4

# export TRANSFORMERS_CACHE=/home/yizhongw/.cache/huggingface
export data_repo="../natural-instructions/"
export SEED=42
# export run_name="${SplitName}_${AL_TYPE}_${AL_ITER}_tk_BS${ExpectBatchSize}_${InstNumPerTask}_lr1e-4_large_def_pos2"
# export model_name_or_path="allenai/tk-instruct-small-def-pos"
export model_name_or_path=output/my_experiment/TLAL/${run_name}
# export model_name_or_path="output/my_experiment/My_tk_BS128_200_lr1e-4_large_label_pos"
# rm -rf ~/.cache/huggingface/datasets/ni_dataset/


port=$(shuf -i25000-30000 -n1)

deepspeed --master_port $port src/run_s2s.py \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path ${model_name_or_path} \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task 0 \
    --max_num_instances_per_eval_task 0 \
    --add_task_name False \
    --add_task_definition True \
    --num_pos_examples 2 \
    --num_neg_examples 0 \
    --add_explanation False \
    --tk_instruct False \
    --data_dir $DataDir \
    --task_dir ${data_repo}tasks \
    --output_dir output/my_experiment/TLAL/${run_name} \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_eval_batch_size 64 \
    --deepspeed ds_configs/stage3.config \
    --bf16 \
    --seed $SEED \
    --pred_remain \
    ${additional_args} \
    --run_name ${run_name}
    