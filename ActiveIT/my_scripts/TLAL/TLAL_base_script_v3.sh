#!/bin/bash
set -x
export TORCH_EXTENSIONS_DIR=/home/ponienkung # refer to https://github.com/pytorch/pytorch/issues/34238
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="$1"
export DataDir=$2
export run_name=$3

# export TRANSFORMERS_CACHE=/home/yizhongw/.cache/huggingface
export data_repo="../natural-instructions/"

export InstNumPerTask="200"
export SEED=42
export ExpectBatchSize=128
export PerDeviceTrainBatchSize=4
# export run_name="${SplitName}_${AL_TYPE}_${AL_ITER}_tk_BS${ExpectBatchSize}_${InstNumPerTask}_lr1e-4_large_def_pos2"
# export model_name_or_path="allenai/tk-instruct-small-def-pos"
export model_name_or_path="google/t5-large-lm-adapt"
# export model_name_or_path="output/my_experiment/My_tk_BS128_200_lr1e-4_large_label_pos"

# Auto cal grad_accum for batch 128
ceildiv(){ echo $((($1+$2-1)/$2)); }
GPU_NUM=$(ceildiv ${#CUDA_VISIBLE_DEVICES} 2)
GradAccum=$((ExpectBatchSize / PerDeviceTrainBatchSize / GPU_NUM))
echo "$GradAccum"

port=$(shuf -i25000-30000 -n1)

# No example, short source length, larger original batch size, 100 to 200 instances, less epochs 10 to 4

deepspeed --master_port $port src/run_s2s.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path ${model_name_or_path} \
    --max_source_length 1024 \
    --max_target_length 128 \
    --generation_max_length 128 \
    --max_num_instances_per_task $InstNumPerTask \
    --max_num_instances_per_eval_task 50 \
    --max_num_instances_per_test_task 100 \
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
    --per_device_train_batch_size $PerDeviceTrainBatchSize \
    --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps $GradAccum \
    --learning_rate 5e-05 \
    --num_train_epochs 6 \
    --lr_scheduler_type constant \
    --warmup_steps 50 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --deepspeed ds_configs/stage2.config \
    --bf16 \
    --load_best_model_at_end \
    --metric_for_best_model rougeL_for_train \
    --save_total_limit 1 \
    --seed $SEED \
    --run_name ${run_name}
    
