export GPUS="0,1,2,3"

# Important Note: For reproduce purpose, we will add the arg --no_update_task_list, which disable the model to update the task list and use the task list we get from our experiments
# The reason why we do this is because the task selection can have huge variance for single run. By using this metric, you can get a closer performance to our experiment results
# If you disable this argument, you should still get similar results when averaging results from multiple(5) random seeds. This can take time though.

# If not reproducing, make sure to remove --no_update_task_list args

# Baseline: Random Sampling
python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type Random --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_10  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type Random --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_20  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type Random --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_30  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type Random --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_40  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type Random --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_50  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list

# # Baseline: Low Perplexity
python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type LowPerplexity --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_10  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type LowPerplexity --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_20  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type LowPerplexity --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_30  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type LowPerplexity --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_40  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type LowPerplexity --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_50  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list

# # Baseline: High Perplexity
python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type HighPerplexity --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_10  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type HighPerplexity --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_20  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type HighPerplexity --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_30  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type HighPerplexity --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_40  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type HighPerplexity --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_50  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --no_update_task_list

# # Proposed: Prompt Uncertainty
python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type FCGRatioPI-NL-I10-TDTE-High-0.2-Mean-abs-bald --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_10  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --perturb_num 10 --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type FCGRatioPI-NL-I10-TDTE-High-0.2-Mean-abs-bald --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_20  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --perturb_num 10 --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type FCGRatioPI-NL-I10-TDTE-High-0.2-Mean-abs-bald --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_30  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --perturb_num 10 --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type FCGRatioPI-NL-I10-TDTE-High-0.2-Mean-abs-bald --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_40  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --perturb_num 10 --no_update_task_list
# python3 my_scripts/TLAL/TLAL_pipeline.py --AL_type FCGRatioPI-NL-I10-TDTE-High-0.2-Mean-abs-bald --gpus $GPUS --split_dir ../natural-instructions/splits/TLAL_Exp0_all_50  --max_iter 5 --fix_cls_gen_ratio 0.356 --base_script my_scripts/TLAL/TLAL_base_script_v4.sh --perturb_num 10 --no_update_task_list
