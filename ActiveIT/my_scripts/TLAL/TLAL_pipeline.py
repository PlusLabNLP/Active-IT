import argparse
import os
import subprocess
import sys
import pathlib
import json
from tqdm import tqdm
from TLAL_utils import *
# from subprocess import Popen, PIPE

def update_task_list(args, prev_data_dir, data_dir, prev_output_examples, algo, chunk_size, fix_cls_gen_ratio = None):
    # When calling this function, we update the task list for next iteration
    train_file = os.path.join(prev_data_dir, "train_tasks.txt")
    dev_file = os.path.join(prev_data_dir, "dev_tasks.txt")
    remain_file = os.path.join(prev_data_dir, "remain_tasks.txt")
    train_task_list = [line.strip() for line in open(train_file, "r").readlines()]
    dev_task_list = [line.strip() for line in open(dev_file, "r").readlines()]
    remain_task_list = [line.strip() for line in open(remain_file, "r").readlines()]
    # Following the algorithm to select #chunk_size of new tasks
    if fix_cls_gen_ratio is None: # Don;t fix the ratio
        if algo=="Random":
            remain_used, remain_unused = RandomSelect(remain_task_list, chunk_size)
        elif algo=="RandomCategory":
            remain_used, remain_unused = RandomSelect(remain_task_list, chunk_size, select_type="category")
        elif algo=="RandomCLS":
            remain_used, remain_unused = RandomSelect(remain_task_list, chunk_size, select_type="cls")
        elif algo=="RandomGEN":
            remain_used, remain_unused = RandomSelect(remain_task_list, chunk_size, select_type="gen")
        elif algo=="LowPerplexity":
            remain_used, remain_unused = PerplexitySelect(args, remain_task_list, chunk_size, prev_output_examples)
        elif algo=="HighPerplexity":
            remain_used, remain_unused = PerplexitySelect(args, remain_task_list, chunk_size, prev_output_examples, select_high=True)
        elif "PI-" in algo: # Perturb Instruct
            remain_used, remain_unused = PI_Std_Select(args, remain_task_list, chunk_size, prev_output_examples)
        else:
            # print(algo == "Random", type(algo), len(algo))
            print(f"Error! Algo {algo} is not defined in the script...")
            exit(0)
    else:
        assert not (algo in ["RandomCLS", "RandomGEN"]), "Error! When fix_cls_gen_ratio is specified, should not specify AL_type as RandomCLS or RandomGEN! The concept contradict each other."
        remain_task_list_cls, remain_task_list_gen = split_cls_gen(remain_task_list)
        cls_num = min(int(chunk_size*fix_cls_gen_ratio), len(remain_task_list_cls))
        gen_num = chunk_size - cls_num
        
        if algo=="Random":
            remain_used_cls, remain_unused_cls = RandomSelect(remain_task_list_cls, cls_num)
            remain_used_gen, remain_unused_gen = RandomSelect(remain_task_list_gen, gen_num)
        elif algo=="RandomCategory":
            remain_used_cls, remain_unused_cls = RandomSelect(remain_task_list_cls, cls_num, select_type="category")
            remain_used_gen, remain_unused_gen = RandomSelect(remain_task_list_gen, gen_num, select_type="category")
        elif algo=="LowPerplexity":
            remain_used_cls, remain_unused_cls = PerplexitySelect(args, remain_task_list_cls, cls_num, prev_output_examples)
            remain_used_gen, remain_unused_gen = PerplexitySelect(args, remain_task_list_gen, gen_num, prev_output_examples)
        elif algo=="HighPerplexity":
            remain_used_cls, remain_unused_cls = PerplexitySelect(args, remain_task_list_cls, cls_num, prev_output_examples, select_high=True)
            remain_used_gen, remain_unused_gen = PerplexitySelect(args, remain_task_list_gen, gen_num, prev_output_examples, select_high=True)
        elif "PI-" in algo: # Perturb Instruct
            remain_used_cls, remain_unused_cls = PI_Std_Select(args, remain_task_list_cls, cls_num, prev_output_examples)
            remain_used_gen, remain_unused_gen = PI_Std_Select(args, remain_task_list_gen, gen_num, prev_output_examples)
        else:
            # print(algo == "Random", type(algo), len(algo))
            print(f"Error! Algo {algo} is not defined in the script...")
            exit(0)
        remain_used = remain_used_cls + remain_used_gen
        remain_unused = remain_unused_cls + remain_unused_gen
    # Write list
    write_train_task_list = train_task_list + remain_used
    write_remain_task_list = remain_unused
    # Write files
    write_train_file = os.path.join(data_dir, "train_tasks.txt")
    write_remain_file = os.path.join(data_dir, "remain_tasks.txt")
    write_file(write_train_file, write_train_task_list)
    write_file(write_remain_file, write_remain_task_list)
    assert len(set(write_train_task_list).intersection(set(write_remain_task_list)))==0, "Error! Train task list have same items as in the Remain task list!!"
    print(f"Done updating data_dir {data_dir}")
        

parser = argparse.ArgumentParser(description='Arguments for running TLAL pipeline')
parser.add_argument('--AL_type', type=str, required=True, default="Random",
                    help='The task selecting strategy for task-level active learning. Default is set to Random.')
parser.add_argument('--gpus', type=str, required=True,
                    help='The gpus to run. For examples 0,1,2,3')
parser.add_argument('--split_dir', type=str, default="../natural-instructions/splits/TLAL_all_10",
                    help='The base split folder containing the task split')
parser.add_argument('--max_iter', type=int, default=None,
                    help='The max number of iter to run')
parser.add_argument('--skip_iter_for_random', type=int, default=None,
                    help='For random, sampling, can skip to a specific iter')
parser.add_argument('--exp_postfix', type=str, default="",
                    help='The postfix for experiment, default to empty str')
# Don't need to specify the following args except for special occasions.
parser.add_argument('--output_dir', type=str, default="output/my_experiment/TLAL",
                    help='The base output folder for TLAL experiment')
parser.add_argument('--chunks', type=int, default=10,
                    help='The number of chunks for active learning')
parser.add_argument('--base_script', type=str, default="my_scripts/TLAL/TLAL_base_script_v4.sh",
                    help='The base script to run for the pipeline')
parser.add_argument('--base_pred_script', type=str, default="my_scripts/TLAL/TLAL_base_pred_script.sh",
                    help='The base script to run prediction for the pipeline')
parser.add_argument('--instance_num', type=int, default=10,
                    help='The number of test instances')
parser.add_argument('--perturb_num', type=int, default=20,
                    help='The number of perturb instructions')
parser.add_argument('--fix_cls_gen_ratio', type=float, default=None,
                    help='Fix the ratio of cls and gen to a certain value')
parser.add_argument('--pred_file_prefix_for_original_def', type=str, default="OriginalDef_",
                    help='The prefix for predicted_file for predicting original task definition on remainging tasks')
parser.add_argument('--no_update_task_list', action='store_true')



def main():
    args = parser.parse_args()
    # Start running pipeline
    # 1. copy dir for the run
    copy_dirs(
        args.split_dir, 
        args.AL_type, 
        args.chunks
    )
    prev_original_def_examples = None
    # Get the chunk_size from the training tasks in init folder
    chunk_size = len([line.strip() for line in open(os.path.join(args.split_dir, "init", "train_tasks.txt"), "r").readlines()])
    # Get the name of the TLAL split
    split_name = pathlib.PurePath(args.split_dir).name
    total_iter = args.max_iter if args.max_iter is not None else args.chunks
    pred_file_name = get_pred_file_name(args.AL_type) # Name of the predicted_example file
    used_label_file = None
    for it in tqdm(range(total_iter)):
        run_name = f"{split_name}_{args.AL_type}_{it}_{args.exp_postfix}"
        data_dir = os.path.join(args.split_dir, f"{args.AL_type}_{it}")
        if it==0 and (not check_process_done(args.output_dir, run_name)): # Check if there're Random_0 experiment already done. If true, make a softlink to it
            Rand_0_name = f"{split_name}_Random_0"
            Rand_0_folder_list = [f for f in os.listdir(args.output_dir) if Rand_0_name in f]
            print(Rand_0_folder_list)
            # assert len(Rand_0_folder_list) <=1, f"Error, There are {len(Rand_0_folder_list)} folders that have name {Rand_0_name} in folder {args.output_dir}"
            if len(Rand_0_folder_list) >= 1: #exist previous file
                Rand_0_folder = Rand_0_folder_list[0]
                if check_process_done(args.output_dir, Rand_0_folder):
                    print(f"{Rand_0_folder} is done, soft link to it.")
                    os.symlink(
                        os.path.abspath(os.path.join(args.output_dir, Rand_0_folder)),
                        os.path.abspath(os.path.join(args.output_dir, run_name)),
                    )
                else:
                    print(f"{Rand_0_folder} is not done...")
        if not check_process_done(args.output_dir, run_name): # Skip if already done
            if it!=0 and (not args.no_update_task_list):
                # After running training, modify the new dir by adding new tasks into the train_tasks.txt
                update_task_list(args, prev_data_dir, data_dir, prev_output_examples, algo=args.AL_type, chunk_size=chunk_size, fix_cls_gen_ratio = args.fix_cls_gen_ratio)
            if args.skip_iter_for_random is not None and (it < args.skip_iter_for_random):
                prev_data_dir = data_dir
                prev_output_examples = []
                # prev_output_example_file = os.path.join(args.output_dir, run_name, pred_file_name)
                # with open(prev_output_example_file, "r") as F:
                #     for line in F.readlines():
                #         prev_output_examples.append(json.loads(line))
                print(f"----- Skip iter: {it}")
                continue
            # For some AL type, will add additional args to training  
            subprocess.check_call(
                [
                    args.base_script, 
                    args.gpus,
                    data_dir, 
                    run_name,
                ], 
                stdout=sys.stdout, 
                stderr=subprocess.STDOUT
            )
            assert check_process_done(args.output_dir, run_name), f"Error. Break at Iter: {it}. No predict results"
        if "PI-" in args.AL_type: # For PI-NL
            HasLabel, InstanceNum, _, _ = parse_PI_uncertainty(args.AL_type)
            if not HasLabel:
                OriginalDefPredFile = f"{args.pred_file_prefix_for_original_def}predicted_examples.jsonl"
                used_label_file = os.path.join(args.output_dir, run_name, OriginalDefPredFile)
        if (it != total_iter-1) and args.AL_type!="Random" and not check_process_done(args.output_dir, run_name, pred_file_name = pred_file_name):
            # Skip if: 
            # (1) the last iter, don't need to predict uncertainty
            # (2) Is Random
            # (3) already done
            # Need to get the prediction of using original task definition first
            if "PI-" in args.AL_type:
                if not HasLabel: # No Label, run prediction
                    # Check already have the prediction file
                    if not check_process_done(args.output_dir, run_name, pred_file_name = OriginalDefPredFile):
                        # Run it
                        pred_args = [
                            f"--max_num_instances_per_test_task {100 if it == 0 else InstanceNum}", # For the first one, since it will be share in folder Random_0 and will only be ran once, we can try to run more data(100) on it.
                            f"--pred_file_prefix {args.pred_file_prefix_for_original_def}",
                        ]
                        additional_args = " ".join(pred_args)
                        subprocess.check_call(
                            [
                                args.base_pred_script, 
                                args.gpus,
                                data_dir, 
                                run_name,
                                additional_args,
                            ], 
                            stdout=sys.stdout, 
                            stderr=subprocess.STDOUT
                        )
                    assert check_process_done(args.output_dir, run_name, pred_file_name = OriginalDefPredFile), f"Error. Break at Iter: {it}. No predicted examples file {pred_file_name}."

            if not check_process_done(args.output_dir, run_name, pred_file_name = pred_file_name):
                # Need to pred remain tasks for uncertainty
                additional_args = get_additional_args(args, args.AL_type)
                if used_label_file is not None:
                    # When specify this argument, the ni_collator will use the generated results in this file for label
                    additional_args += " " + f"--used_label_file {used_label_file}"
                subprocess.check_call(
                    [
                        args.base_pred_script, 
                        args.gpus,
                        data_dir, 
                        run_name,
                        additional_args,
                    ], 
                    stdout=sys.stdout, 
                    stderr=subprocess.STDOUT
                )
            assert check_process_done(args.output_dir, run_name, pred_file_name = pred_file_name), f"Error. Break at Iter: {it}. No predicted examples file {pred_file_name}."

        if (it != total_iter-1):
            prev_data_dir = data_dir
            prev_output_example_file = os.path.join(args.output_dir, run_name, pred_file_name)
            prev_output_examples = []
            with open(prev_output_example_file, "r") as F:
                for line in F.readlines():
                    prev_output_examples.append(json.loads(line))
        # if used_label_file is not None:
        #     print("---------- Set prev_original_def_examples")
        #     prev_original_def_examples = []
        #     with open(used_label_file, "r") as F:
        #         for line in F.readlines():
        #             prev_original_def_examples.append(json.loads(line))
        
        
if __name__ == "__main__":
    main()