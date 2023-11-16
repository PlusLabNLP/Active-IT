import shutil
import os
import json
import numpy as np
import torch
import random

def is_cls(meta_data, task):
    if meta_data[task+".json"]['DefHasLabel'] == 'ExactMentioned':
        return True
    else:
        return False

def split_cls_gen(task_list):
    # Load MetaData
    meta_data_file = "task_metadata.json"
    meta_data = json.load(open(meta_data_file, "r"))
    cls_list = []
    gen_list = []
    for task in task_list:
        if is_cls(meta_data, task):
            cls_list.append(task)
        else:
            gen_list.append(task)
    return cls_list, gen_list

def copy_dirs(split_dir, AL_type, chunks):
    # Copy following dir
    dir_src = os.path.join(split_dir, "init")
    for i in range(chunks):
        dir_dst = os.path.join(split_dir, f"{AL_type}_{i}")
        try:
            shutil.copytree(dir_src, dir_dst)
        except:
            print(f"Dir {dir_dst} already exist, don't copy")
    print(f"Done copying dir from {dir_src} to {dir_dst}")

def write_file(file_name, task_list):
    print(f"Write len {len(task_list)} to {file_name}")
    with open(file_name, "w") as F:
        for task in task_list:
            F.write(task+"\n")

def parse_PI_uncertainty(AL_type):
    args = AL_type.split("-")
    if args[1] =="HL":
        HasLabel = True
    else:
        HasLabel = False
    InstanceNum = int(args[2][1:])
    PerturbMethod = args[3]
    SelectHigh = args[4] == "High"
    return HasLabel, InstanceNum, PerturbMethod, SelectHigh

def get_additional_args(args, AL_type):
    if "Perplexity" in AL_type:
        pred_args = [
            "--uncertainty perplexity",
            f"--max_num_instances_per_test_task {args.instance_num}",
            f"--perturb_num 0",
        ]
        return " ".join(pred_args)
    elif "PI-" in AL_type: # don;t use default instance num
        if "bald" in AL_type:
            perturb_num = args.perturb_num+1
        else:
            perturb_num = args.perturb_num
        HasLabel, InstanceNum, PerturbMethod, SelectHigh = parse_PI_uncertainty(AL_type)
        pred_args = [
            f"--uncertainty {AL_type}",
            f"--max_num_instances_per_test_task {InstanceNum}",
            f"--perturb_num {perturb_num}",
            f"--perturb_inst {PerturbMethod}",
            "--skip_generate",
        ]
        if PerturbMethod in ["TD", "ALL", "TDTE"]:
            DropRate = AL_type.split("-")[5]
            pred_args += [f"--perturb_def_drop_rate {DropRate}"]
            if "Mask" in AL_type:
                pred_args += ["--perturb_def_mask"]
        return " ".join(pred_args)
def get_pred_file_name(AL_type):
    original = "predicted_examples.jsonl"
    if "Random" in AL_type:
        return original
    elif "Perplexity" in AL_type:
        return f"perplexity_{original}"
    elif "PI-" in AL_type:
        return f"{AL_type.replace('-High', '').replace('-Low', '')}_{original}"

def check_process_done(output_dir ,run_name, pred_file_name = None):
    # by checking whether the predict_results.json exist
    if pred_file_name is None: # Check predicted examples
        pred_file = os.path.join(output_dir, run_name, "predict_results.json")
        print(f"------- {pred_file} exist? --> {os.path.isfile(pred_file)}")
        return os.path.isfile(pred_file)
    else:
        pred_file = os.path.join(output_dir, run_name, pred_file_name)
        print(f"------- {pred_file} exist? --> {os.path.isfile(pred_file)}")
        return os.path.isfile(pred_file)


def get_task_perplexity(prev_output_examples, remain_task_list, example_num):
    print("--------------------------------------")
    uncertainty_list_dict = {}
    uncertainty_dict = {}
    for example in prev_output_examples:
        task = example['Task']
        if task not in remain_task_list:
            # skip
            continue 
        uncertainty = example['uncertainty']
        if task not in uncertainty_list_dict:
            uncertainty_list_dict[task] = [uncertainty]
        else:
            uncertainty_list_dict[task].append(uncertainty)
    assert len(uncertainty_list_dict) == len(remain_task_list), f"Error! The uncertainty dict length not equal to remain task list. {len(uncertainty_list_dict)} != {len(remain_task_list)}"
    for task, uncertainty_list in uncertainty_list_dict.items():
        assert len(uncertainty_list) == example_num, f"Example num of task {task} is {len(uncertainty_list)} instead of {example_num}."
        uncertainty_dict[task] = sum(uncertainty_list)/example_num
    return uncertainty_dict

def get_task_PI_StdOrMean(prev_output_examples, remain_task_list, example_num):
    uncertainty_list_dict = {}
    uncertainty_mean_dict = {}
    uncertainty_std_dict = {}
    for example in prev_output_examples:
        task = example['Task']
        if task not in remain_task_list:
            # skip
            continue 
        uncertainty = example['uncertainty']
        if task not in uncertainty_list_dict:
            uncertainty_list_dict[task] = [uncertainty]
        else:
            uncertainty_list_dict[task].append(uncertainty)
    assert len(uncertainty_list_dict) == len(remain_task_list), f"Error! The uncertainty dict length not equal to remain task list. {len(uncertainty_list_dict)} != {len(remain_task_list)}"
    for task, uncertainty_list in uncertainty_list_dict.items():
        assert len(uncertainty_list) == example_num, f"Example num of task {task} is {len(uncertainty_list)} instead of {example_num}."
        # The current uncertainty is the loss. We want to use the prob but not logprob, so we will use torch.exp here
        uncertainty_list = np.exp(-np.array(uncertainty_list))
        uncertainty_mean_dict[task] = np.mean(np.array(uncertainty_list))
        uncertainty_std_dict[task] = np.std(np.array(uncertainty_list))
    return uncertainty_mean_dict, uncertainty_std_dict

def get_task_PI_StdOrMean_OriginalDef(prev_output_examples, remain_task_list, example_num, abs_diff = False):
    print("---------------------- Have Original Def Examples! ----------------")
    print(f"Absolute Diff: {abs_diff}")
    id_to_sentence_prob_dict  = {}
    for example in prev_output_examples:
        perturb_id = int(example['id'].split("-")[-1])
        id = "-".join(example['id'].split("-")[:-1])
        task = example['Task']
        if task not in remain_task_list or (perturb_id != 0):
            continue 
        sentence_prob = np.exp(-example['uncertainty'])
        id_to_sentence_prob_dict[id] = sentence_prob
    uncertainty_list_dict = {}
    original_def_prob_dict = {}
    uncertainty_mean_dict = {}
    uncertainty_std_dict = {}
    for example in prev_output_examples:
        perturb_id = int(example['id'].split("-")[-1])
        id = "-".join(example['id'].split("-")[:-1])
        task = example['Task']
        if task not in remain_task_list or (perturb_id == 0):
            # skip
            continue 
        sentence_prob = np.exp(-example['uncertainty'])
        sentence_prob -= id_to_sentence_prob_dict[id] #calculate the diff
        if abs_diff:
            sentence_prob = abs(sentence_prob)
        if task not in uncertainty_list_dict:
            uncertainty_list_dict[task] = [sentence_prob]
        else:
            uncertainty_list_dict[task].append(sentence_prob)
        if task not in original_def_prob_dict:
            original_def_prob_dict[task] = [id_to_sentence_prob_dict[id]]
        else:
            original_def_prob_dict[task].append(id_to_sentence_prob_dict[id])
    for task, prob_list in original_def_prob_dict.items():
        uncertainty_std_dict[task] = np.mean(np.array(prob_list))
    assert len(uncertainty_list_dict) == len(remain_task_list), f"Error! The uncertainty dict length not equal to remain task list. {len(uncertainty_list_dict)} != {len(remain_task_list)}"
    for task, uncertainty_list in uncertainty_list_dict.items():
        assert len(uncertainty_list) == example_num, f"Example num of task {task} is {len(uncertainty_list)} instead of {example_num}."
        uncertainty_list = np.array(uncertainty_list)
        uncertainty_mean_dict[task] = np.mean(np.array(uncertainty_list))
        # uncertainty_std_dict[task] = np.std(np.array(uncertainty_list))
        # Mean is the prompt uncertainty, std is the sentence prob
    return uncertainty_mean_dict, uncertainty_std_dict


# Task selection algos
def RandomSelect(remain_task_list, chunk_size, select_type="all"):
    random.shuffle(remain_task_list)
    if select_type == "all":
        return remain_task_list[:chunk_size], remain_task_list[chunk_size:]
    elif select_type == "category":
        print("------------ In Category")
        meta_data_file = "task_metadata.json"
        meta_data = json.load(open(meta_data_file, "r"))
        used_tasks = []
        unused_tasks = []
        selected_categories = []
        # First try to select one task from all category
        for task in remain_task_list:
            category = meta_data[task+".json"]['Categories'][0]
            if (len(used_tasks) == chunk_size) or (category in selected_categories):
                unused_tasks.append(task)
            else:
                used_tasks.append(task)
                selected_categories.append(category)
        # if still need to random sample some tasks
        if len(used_tasks) < chunk_size:
            to_random_sample_num = chunk_size - len(used_tasks)
            random.shuffle(unused_tasks)
            used_tasks += unused_tasks[:to_random_sample_num]
            unused_tasks = unused_tasks[to_random_sample_num:]
        return used_tasks, unused_tasks
    else:
        meta_data_file = "task_metadata.json"
        meta_data = json.load(open(meta_data_file, "r"))
        cls_tasks = []
        gen_tasks = []
        for task in remain_task_list:
            if meta_data[task+".json"]['DefHasLabel'] == "ExactMentioned":
                cls_tasks.append(task)
            else:
                gen_tasks.append(task)
        if select_type == "cls":
            if len(cls_tasks) >= chunk_size: # have enough tasks
                use_task_list = cls_tasks[:chunk_size]
            else: # Not enough tasks
                use_task_list = cls_tasks + gen_tasks[:chunk_size - len(cls_tasks)]
        if select_type == "gen":
            if len(gen_tasks) >= chunk_size: # have enough tasks
                use_task_list = gen_tasks[:chunk_size]
            else: # Not enough tasks
                use_task_list = gen_tasks + cls_tasks[:chunk_size - len(gen_tasks)]
        return use_task_list, [task for task in remain_task_list if task not in use_task_list]


def PerplexitySelect(args, remain_task_list, chunk_size, prev_output_examples, select_high = False):
    # sort_desc == True --> Desc Perplexity, Select high perplexity
    uncertainty_dict = get_task_perplexity(prev_output_examples, remain_task_list, example_num = args.instance_num)
    task_uncertainty_sorted = sorted(uncertainty_dict.items(), key=lambda x:x[1], reverse=select_high)
    task_sorted = [task_uncertainty[0] for task_uncertainty in task_uncertainty_sorted]
    return task_sorted[:chunk_size], task_sorted[chunk_size:]

def PI_Std_Select(args, remain_task_list, chunk_size, prev_output_examples):
    HasLabel, InstanceNum, PerturbMethod, SelectHigh = parse_PI_uncertainty(args.AL_type)
    example_num_per_task = InstanceNum * args.perturb_num
    # sort_desc == True --> Desc Perplexity, Select high perplexity
    if "bald" in args.AL_type:
        uncertainty_mean_dict, uncertainty_std_dict = get_task_PI_StdOrMean_OriginalDef(prev_output_examples, remain_task_list, example_num=example_num_per_task, abs_diff = ("abs" in args.AL_type))
    else:
        uncertainty_mean_dict, uncertainty_std_dict = get_task_PI_StdOrMean(prev_output_examples, remain_task_list, example_num=example_num_per_task)
        
    uncertainty_mean_dict_sorted = sorted(uncertainty_mean_dict.items(), key=lambda x:x[1]) # Low to High
    uncertainty_std_dict_sorted = sorted(uncertainty_std_dict.items(), key=lambda x:x[1]) # Low to High
    mean_sorted_tasks = [task_uncertainty[0] for task_uncertainty in uncertainty_mean_dict_sorted]
    std_sorted_tasks = [task_uncertainty[0] for task_uncertainty in uncertainty_std_dict_sorted]
    if "Mean" in args.AL_type:
        if SelectHigh: #High to Low
            mean_sorted_tasks.reverse()
        return mean_sorted_tasks[:chunk_size], mean_sorted_tasks[chunk_size:]
    elif "MergeRank" in args.AL_type: # Merge by rank
        # Should look like this --> *MergeRank_H_L_0.2_PI-*
        _, MeanOrder, StdOrder, MeanRate, _ = args.AL_type.split("PI-")[0].split("_")
        MeanRate = float(MeanRate)
        assert MeanOrder in ["H", "L"], f"Error! MeanOrder should be \'H\' or \'L\' !!! Instead get {MeanOrder}"
        assert StdOrder in ["H", "L"], f"Error! StdOrder should be \'H\' or \'L\' !!! Instead get {MeanOrder}"
        assert MeanRate <= 1 and MeanRate >=0, f"Error!! MeanRate should be between (0,1) but instead get {MeanRate}"
        if MeanOrder == "H":
            mean_sorted_tasks.reverse()
        if StdOrder == "H":
            std_sorted_tasks.reverse() #High to Low
        MeanSize = int(chunk_size*MeanRate//1)
        StdSize = int(chunk_size - MeanSize)
        # First select MeanSize tasks into used_tasks, then add the rest
        used_tasks = mean_sorted_tasks[:MeanSize]
        for task in std_sorted_tasks:
            if len(used_tasks) == chunk_size:
                break
            if not task in used_tasks:
                used_tasks.append(task)
        remain_tasks = list(set(mean_sorted_tasks) - set(used_tasks))
        assert len(used_tasks + remain_tasks) == len(mean_sorted_tasks), f"Error! used_tasks {len(used_tasks)} + remain_tasks {len(remain_tasks)} != original_remaining_tasks {len(mean_sorted_tasks)}"
        return used_tasks, remain_tasks
    elif "SumRank" in args.AL_type: # Merge by rank
        if "ReverseSTD" in args.AL_type:
            std_sorted_tasks.reverse() # High to low
        task_rank_dict = {}
        for i, task in enumerate(mean_sorted_tasks): #Low to High
            task_rank_dict[task] = i
        for i, task in enumerate(std_sorted_tasks): #Low to High
            task_rank_dict[task] += i
        task_rank_sorted = sorted(task_rank_dict.items(), key=lambda x:x[1]) # Low to High
        rank_sorted_tasks = [task_rank[0] for task_rank in task_rank_sorted] #Low rank to high rank
        if SelectHigh: # Select the one with highest rank
            rank_sorted_tasks.reverse() #High to Low
        return rank_sorted_tasks[:chunk_size], rank_sorted_tasks[chunk_size:]
    else:
        if SelectHigh:
            std_sorted_tasks.reverse() #High to Low
        return std_sorted_tasks[:chunk_size], std_sorted_tasks[chunk_size:]
    