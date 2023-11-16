import json
import os
import shutil
import random
import numpy as np
from matplotlib import transforms
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import seaborn as sns
from ipynb.fs.defs.get_results import get_scores
from tqdm import tqdm
import scipy.stats as ss

# ---------------------------- Column 1

# Return True id a task is classification(cls) task
def is_cls(meta_data, task):
    if meta_data[task+".json"]['DefHasLabel'] == 'ExactMentioned':
        return True
    else:
        return False

# Function to write file
def write_file(file_name, task_list):
    with open(file_name, "w") as F:
        for task in task_list:
            F.write(task+"\n")

"""
Function to write split_dir, includes train_tasks.txt, remain_tasks.txt, test_tasks.txt, dev_tasks.txt and excluded_tasks.txt
For ActiveIT with mutliple iterations, will write a split_dir for each of the iteration.
For first iter, we will random sample a chunk(756//11 = 68) of tasks as training tasks. And the later iter split_dir will be exact same as the first iter.
"""
def write_TLAL_split(
        # The meta_data file
        meta_data,
        # The list of all training tasks, which is a task pool
        train_task_list,
        # The list of testing tasks
        test_task_list,
        # The random seed to set
        seed,
        # The total Active IT iterations. Which can help us get the number of tasks for each chunk
        chunks = 10,
        # Whether to sample cls and gen tasks following a 50%/50% ratio for the dev set
        balance_dev_CLSGEN = False,
        # the name of the created split. When conducting experiments for multiple round and you want to try different task pool, this can be useful
        split_name = None
    ):
    
    # Sanity Check
    assert type(seed)==int, f"Seed should be a integer instead of type(seed) = {type(seed)}"
    
    # Prepare
    random.seed(seed)
    
    # Try to create TLAL dir, in it will have chunks of dirs, id from 0 to chunks-1 
    split_path = "../natural-instructions/splits/"
    TLAL_dir = os.path.join(split_path, f"TLAL_{split_name}_{seed}" if split_name is not None else f"TLAL_{seed}")
    try:
        os.mkdir(TLAL_dir)
        print(f"Succesfully create dir {TLAL_dir}")
    except:
        print(f"Didn't create dir {TLAL_dir}, maybe it already exists")
    
    # Create init dir and copy the excluded_tasks.txt
    init_dir = os.path.join(TLAL_dir, "init")
    try:
        os.mkdir(init_dir)
        print(f"Succesfully create dir {init_dir}")
    except:
        print("Didn't create dir {init_dir}, maybe it already exists")
    exc_src = os.path.join(split_path, "default_dev", "excluded_tasks.txt")
    exc_dst = os.path.join(init_dir, "excluded_tasks.txt")
    shutil.copyfile(exc_src, exc_dst)
    
    # Now adding the tasks file into init/ dir
    train_file = os.path.join(init_dir, "train_tasks.txt")
    remain_file = os.path.join(init_dir, "remain_tasks.txt")
    test_file = os.path.join(init_dir, "test_tasks.txt")
    dev_file = os.path.join(init_dir, "dev_tasks.txt")
    
    test_list = test_task_list.copy()
    train_list_shuffle = train_task_list.copy()
    
    # Shuffle the train_task_list(Task Pool), we will later get 68 random tasks from it as the first iter training tasks
    random.shuffle(train_list_shuffle)
    
    # 756//(10+1) = 68
    chunk_task_num = len(train_list_shuffle) // (chunks+1)
    
    # Get list of tasks: [train_tasks, remain_tasks, test_tasks, dev_tasks]
    if balance_dev_CLSGEN:
        # We sample 68/2 cls tasks and 68/2 gen tasks for dev set
        chunk_task_num_cls = chunk_task_num//2
        chunk_task_num_gen = chunk_task_num - chunk_task_num_cls
        # Get the cls and gen task list
        train_list_shuffle_cls = [task for task in train_list_shuffle if is_cls(meta_data ,task)]
        train_list_shuffle_gen = [task for task in train_list_shuffle if not is_cls(meta_data, task)]
        print(f"Number of total CLS tasks: {len(train_list_shuffle_cls)}, Number of total Gen tasks: {len(train_list_shuffle_gen)}")
        
        # Get dev tasks
        dev_task_list = train_list_shuffle_cls[:chunk_task_num_cls] + train_list_shuffle_gen[:chunk_task_num_gen]
        # Get train tasks
        task_list_shuffle_wo_val = train_list_shuffle_cls[chunk_task_num_cls:] + train_list_shuffle_gen[chunk_task_num_gen:]
        random.shuffle(task_list_shuffle_wo_val)
        train_list = task_list_shuffle_wo_val[:chunk_task_num]
        # Get remain tasks
        remain_list = task_list_shuffle_wo_val[chunk_task_num:]
        # Set both dev tasks and test tasks = dev tasks + test_tasks
        dev_list = test_list + dev_task_list
        test_list = dev_list # 
    else:
        train_list = train_list_shuffle[:chunk_task_num]
        remain_list = train_list_shuffle[chunk_task_num*2:]
        dev_list = test_list + train_list_shuffle[chunk_task_num : chunk_task_num*2]
        test_list = dev_list # 
        
    # Write all files
    write_file(train_file, train_list)
    write_file(dev_file, dev_list)
    write_file(test_file, test_list)
    write_file(remain_file, remain_list)

    print("Done creating TLAL dir... Now printing the folder structure")
    os.system(f"tree {TLAL_dir}")

# ---------------------------- Column 2

def plot_line_graph_var(
    subplot,
    print_curve_X_list, 
    print_curve_Y_list, 
    print_curve_name_list, 
    Title="", 
    XLabel="", 
    YLabel="", 
    X_MIN_MAX = None, 
    Y_MIN_MAX = None,
    AdditionalScore = None
    ):
    figure(figsize=(9, 6))
    
    if len(print_curve_X_list) > 6:
        cm = plt.get_cmap('gist_rainbow')
        is_rainbow = True
    else:
        cm = plt.get_cmap('tab10')
        is_rainbow = False
    NUM_COLORS = len(print_curve_X_list)
    custom_lines = []
    for i, (X, Y, curve_name) in enumerate(zip(print_curve_X_list, print_curve_Y_list, print_curve_name_list)):
        try:
            Y_med = np.average(np.array(Y), axis=0)
            Y_max = np.max(np.array(Y), axis=0)
            Y_min = np.min(np.array(Y), axis=0)
            Y_std = np.std(np.array(Y), axis=0)/2
            if len(X) != len(Y_med):
                min_len = min(len(X), len(Y_med))
                X = X[:min_len]
                Y_med = Y_med[:min_len]
                Y_max = Y_max[:min_len]
                Y_min = Y_min[:min_len]
            # subplot.fill_between(X, Y_min, Y_max, color=cm(1.*i/NUM_COLORS), alpha=0.3)
            
            color = cm(1.*i/NUM_COLORS) if is_rainbow else cm(i)
            custom_lines.append(Line2D([0], [0], color=color, lw=4))
            # Stdev
            # transform = matplotlib.transforms.Affine2D().translate(0, -0.01*(NUM_COLORS//2) + 0.01*i) + subplot.transData
            # Slightly shift X axis
            shift_base = X[0]/30
            shift_offset = 0 if NUM_COLORS %2 != 0 else shift_base/2
            shift_X = np.array(X) -shift_base*(NUM_COLORS//2) + shift_base*i + shift_offset
            if "Test" in Title and i==0:
                subplot.errorbar(X, Y_med, Y_std, linestyle="none", ecolor="black", capsize=3, alpha=.5)
            subplot.plot(X, Y_med, marker='o', color=color) #linewidth=3
        except Exception as e:
            print(f"Cannot plot {curve_name} due to the error: {e}")
    # If add fully-trained score
    if AdditionalScore is not None:
        subplot.axhline(y = AdditionalScore, color = 'black', linestyle = '--')
        custom_lines.append(Line2D([0], [0], color = 'black', linestyle = '--', lw=4))
        print_curve_name_list.append("Fully Trained (680 Tasks)")
    
    # if print_curve_name_list is not None:
    #     subplot.legend(custom_lines, print_curve_name_list)
    if X_MIN_MAX is not None or Y_MIN_MAX is not None:
        # ax = plt.gca()
        subplot.set_xlim(X_MIN_MAX)
        subplot.set_ylim(Y_MIN_MAX)
    # else:
    #     # ax = plt.gca()
    #     subplot.set_xlim([None, None])
    #     y_min = int(min(Y_min)//1)
    #     # print([y_min, y_min+8])
    #     subplot.set_ylim([y_min, y_min+12])
    subplot.set_title(Title, fontsize=14)
    subplot.set_xticks(print_curve_X_list[0]) 
    subplot.set_xticklabels(print_curve_X_list[0], fontsize=12)
    # subplot.xlabel(XLabel, fontsize=10)
    # subplot.ylabel(YLabel, fontsize=10)
    return custom_lines, print_curve_name_list

def get_row_column(sub_fig_num):
    max_col_num = 4
    for i in range(1, 10):
        if i*4 >= sub_fig_num:
            row = i
            break
    for i in range(1, 10):
        if i*row >= sub_fig_num:
            col=i
            break
    return row, col
def plot_line_graph_var_multiple(*args):
    sub_fig_num = len(args)
    row, col = get_row_column(sub_fig_num)
    print(row, col)
    fig, axs = plt.subplots(
        row, 
        col, 
        figsize=(24, 4*row), 
        sharey='col', 
        # sharey='row'
    )
    # plt.setp(axs, xticks=[i for i in range(len(args[0][0]))], xticklabels=args[0][0])
    if row == 1:
        axs = [axs]
    # for r in range(row):
    #     # Hide Y labels
    #     for ax in axs[r][1:]:
    #         plt.setp(ax.get_yticklabels(), visible=False)
    for i in range(sub_fig_num):
        plot_args = [axs[i//col][i%col]]+args[i]
        custom_lines, print_curve_name_list = plot_line_graph_var(*plot_args)
        legend = fig.legend(custom_lines, print_curve_name_list, loc='lower center', ncol=len(custom_lines), columnspacing=2, bbox_to_anchor=(0.5,-0.11), fontsize=14, frameon=True, borderpad=0.8)
        legend.get_frame().set_alpha(1)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("darkgray")
    plt.show()


def plot_line_graph_var_custom_v1(*args):
    FullyTrainedScore = np.mean([
        [50.51, 57.03, 44.43, 50.86, 58.66, 43.04], 
        [48.96, 53.35, 44.87, 48.24, 59.76, 36.47], 
        [49.41, 53.99, 45.13, 48.28, 54.86, 41.71], 
        [49.39, 54.0, 45.09, 47.52, 59.68, 35.27]
        # [47.7, 51.12, 44.52, 46.37, 52.53, 40.22],
        # [47.09, 49.37, 44.97, 47.22, 57.19, 36.98],
        # [46.46, 50.64, 42.55, 46.35, 59.27, 33.34]
    ], axis = 0)
    plt.style.use('seaborn-v0_8-darkgrid')
    # [],
    # [['Validation -- Classification'], ['Validation -- Generative']]
    grids = [['Test -- Overall',  "E0", 'Validation -- Overall'],
             [[['Test -- Classification', 'Test -- Generative']], "E1", [['Validation -- Classification', 'Validation -- Generative']]]]
    grids_flat = ['Test -- Overall', 'Test -- Classification', 'Test -- Generative', 'Validation -- Overall',  'Validation -- Classification', 'Validation -- Generative']
    
    fig, axd = plt.subplot_mosaic(
        grids, 
        figsize=(14, 7.7),
        layout="constrained",
        gridspec_kw=dict(height_ratios=[1.2, 0.8], width_ratios=[1, 0.07, 1], hspace=0.1)
    )
    # Hide subplot
    axd["E0"].axis('off')
    axd["E1"].axis('off')
    # plt.setp(axs, xticks=[i for i in range(len(args[0][0]))], xticklabels=args[0][0])
    # for r in range(row):
    #     # Hide Y labels
    #     for ax in axs[r][1:]:
    #         plt.setp(ax.get_yticklabels(), visible=False)
    for i, grid_name in enumerate(grids_flat):
        plot_args = [axd[grid_name]]+args[i]
        custom_lines, print_curve_name_list = plot_line_graph_var(*plot_args, AdditionalScore=FullyTrainedScore[i])
    
    legend = fig.legend(custom_lines, print_curve_name_list, loc='lower center', ncol=len(custom_lines), columnspacing=2, bbox_to_anchor=(0.5,-0.11), fontsize=14, frameon=True, borderpad=0.8)
    legend.get_frame().set_alpha(1)
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("darkgray")
    fig.suptitle("Natural Instruction V2 (NIV2) Results", fontsize=20)
    fig.supxlabel("Number of Trainging Tasks", fontsize=16)
    fig.supylabel("Rouge-L Score", fontsize=16)
    # fig.savefig("TLAL_NIV2.pdf", bbox_inches='tight')
    plt.show()

# Function to plot all AL scores

def load_meta_data():
    meta_data_file = "task_metadata.json"
    return json.load(open(meta_data_file, "r"))

def get_split_metric(meta_data, split_name, metric, chunk_task_num, prefix="predict_rougeL_for_"):
    if "cls" in metric:
        metric_list = [prefix + line.strip() for line in open(f"../natural-instructions/splits/{split_name}/init/dev_tasks.txt", "r").readlines()[-chunk_task_num:] if is_cls(meta_data, line.strip())]
    elif "gen" in metric:
        metric_list = [prefix + line.strip() for line in open(f"../natural-instructions/splits/{split_name}/init/dev_tasks.txt", "r").readlines()[-chunk_task_num:] if not is_cls(meta_data, line.strip())]
    else:
        metric_list = [prefix + line.strip() for line in open(f"../natural-instructions/splits/{split_name}/init/dev_tasks.txt", "r").readlines()[-chunk_task_num:]]
    return metric_list

def plot_AL_score(metric_list, AL_type_list, split_name_list, max_iter = None, chunks=10, special_postfix="", plot_custom = None, AL_type_name = None, output_dir = "output/TLAL"):
    meta_data = load_meta_data()
    if max_iter is None:
        max_iter=chunks
    chunk_task_num = len([line for line in open(f"../natural-instructions/splits/{split_name_list[0]}/init/train_tasks.txt", "r")])
    all_args = []
    X = [(i+1)*chunk_task_num for i in range(max_iter)]
    for metric in metric_list:
        Y = []
        if AL_type_name is None:
            Y_Name = AL_type_list
        else:
            assert len(AL_type_name) == len(AL_type_list), "Error!"
            Y_Name = AL_type_name
        for AL_type in AL_type_list:
            AL_type_scores = []
            for split_name in split_name_list:
                model_list = []
                for iter in range(max_iter):
                    model_name = f"{split_name}_{AL_type}_{iter}_{special_postfix}"
                    model_list.append(model_name)
                if "predict_rougeL_for_test" in metric:    
                    all_iter_scores = get_scores(
                        [metric],
                        model_list,
                        output_dir = output_dir,
                        return_full_results = False,
                        metric_avg=False
                    )
                    metric_name = metric
                else:
                    split_metric_list = get_split_metric(meta_data, split_name, metric, chunk_task_num)
                    all_iter_scores = get_scores(
                        split_metric_list,
                        model_list,
                        output_dir = output_dir,
                        return_full_results = False,
                        metric_avg=True
                    )
                    metric_name = "avg_score"
                scores = [all_iter_scores[model][metric_name] for model in model_list if all_iter_scores[model][metric_name] is not None]
                AL_type_scores.append(scores)
            Y.append(AL_type_scores)
            # all_X.append(X)
        
        if "test" in metric:
            split = "Test"
        else:
            split = "Validation"
        if "cls" in metric:
            task_type = "Classification"
        elif "gen" in metric:
            task_type = "Generative"
        else:
            task_type = "Overall"
        
        metric_name = f"{split} -- {task_type}"
        
        all_args.append([
            [X]*len(Y), #print_curve_X_list
            Y, #print_curve_Y_list
            Y_Name, #print_curve_name_list
            metric_name, # f"Scores of {metric}", #Title
            "Number of Tasks", #XLabel
            "Rouge-L" #YLabel
        ])
    if plot_custom is None:
        plot_line_graph_var_multiple(*all_args)
    elif plot_custom == "v1":
        plot_line_graph_var_custom_v1(*all_args)
    else:
        assert False, f"Error plot_custom value = {plot_custom}"

