import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D
import json
import os
from tqdm import tqdm
import scipy.stats as ss

plt.style.use('seaborn-v0_8-whitegrid')

# Get Uncertainty

def get_uncertainty_mean_var(pred_data):
    task_uncertainty_list_dict = {}
    for pred in pred_data:
        task = pred['Task']
        uncertainty = pred['uncertainty']
        if task in task_uncertainty_list_dict:
            task_uncertainty_list_dict[task].append(uncertainty)
        else:
            task_uncertainty_list_dict[task] = [uncertainty]
    assert len(set([len(uncertainty_list) for task, uncertainty_list in task_uncertainty_list_dict.items()])) == 1, "Different number of prediction in each task"
    task_uncertainty_Mean_dict = {}
    task_uncertainty_Var_dict = {}
    for task, uncertainty_list in task_uncertainty_list_dict.items():
        uncertainty_list = np.exp(-np.array(uncertainty_list))
        task_uncertainty_Mean_dict[task] = np.mean(uncertainty_list)
        task_uncertainty_Var_dict[task] = np.std(uncertainty_list)
    return task_uncertainty_Mean_dict, task_uncertainty_Var_dict

def get_uncertainty_mean_var_bald(pred_data):
    id_to_uncertainty = {}
    task_uncertainty_list_dict_for_Mean = {}
    task_uncertainty_list_dict_for_STD = {}
    for pred in pred_data:
        task = pred['Task']
        uncertainty = np.exp(-pred['uncertainty'])
        perturb_id = int(pred['id'].split("-")[-1])
        id = "-".join(pred['id'].split("-")[:-1])
        if perturb_id == 0:
            if task in task_uncertainty_list_dict_for_Mean:
                task_uncertainty_list_dict_for_Mean[task].append(uncertainty)
            else:
                task_uncertainty_list_dict_for_Mean[task] = [uncertainty]
            id_to_uncertainty[id] = uncertainty
        else:
            uncertainty_diff = abs(uncertainty - id_to_uncertainty[id])
            if task in task_uncertainty_list_dict_for_STD:
                task_uncertainty_list_dict_for_STD[task].append(uncertainty_diff)
            else:
                task_uncertainty_list_dict_for_STD[task] = [uncertainty_diff]
    assert len(set([len(uncertainty_list) for task, uncertainty_list in task_uncertainty_list_dict_for_Mean.items()])) == 1, "Different number of prediction(MEAN) in each task"
    assert len(set([len(uncertainty_list) for task, uncertainty_list in task_uncertainty_list_dict_for_STD.items()])) == 1, "Different number of prediction(STD) in each task"
    task_uncertainty_Mean_dict = {}
    task_uncertainty_Var_dict = {}
    for task, uncertainty_list in task_uncertainty_list_dict_for_Mean.items():
        task_uncertainty_Mean_dict[task] = np.mean(uncertainty_list)
    for task, uncertainty_list in task_uncertainty_list_dict_for_STD.items():
        task_uncertainty_Var_dict[task] = np.mean(uncertainty_list)
    return task_uncertainty_Mean_dict, task_uncertainty_Var_dict

def get_selected_tasks(split, uncertainty, run):
    init_train_file = f"../natural-instructions/splits/{split}/init/train_tasks.txt"
    chunk_size = len([line.strip() for line in open(init_train_file, "r").readlines()])
    train_file = f"../natural-instructions/splits/{split}/{uncertainty}_{run}/train_tasks.txt"
    try:
        train_task_list = [line.strip() for line in open(train_file, "r").readlines()]
        if len(train_task_list) != (run+1)*chunk_size: #Error or not processed
            print(f"Error! len(train_task_list) != (run+1)*chunk_size --> {len(train_task_list)} {(run+1)*chunk_size}")
            selected_tasks = None
        else:
            selected_tasks = train_task_list[-chunk_size:]
    except Exception as e:
        print(e)
        selected_tasks = None
    return selected_tasks
    

# Plot 
def forward(x):
    return x**(1/3)
def inverse(x):
    return x**3

# Function to plot scatter for one subgraph
def plot_scatter_main(
    subplot,
    X_list,
    Y_list,
    Label_list,
    Title="",
    CM = 'tab10',
    ):
    # custom_legends = []
    all_colors = []
    subplot.set_xscale('function', functions=(forward, inverse))
    cm = plt.get_cmap(CM)
    for i,(X, Y, Label) in enumerate(zip(X_list, Y_list, Label_list)):
        new_x = []
        new_y = []
        for x, y in zip(X,Y):
            if x < 0.25 or True:
                new_x.append(x)
                new_y.append(y)
        if "All" in Label:
            color = "darkgray"
            # marker = "."
        elif i ==1:
            color = cm(0)
            # marker = "s"
        else:
            color = cm(i)
        marker = "o"
        subplot.scatter(new_x, new_y, s=80, marker=marker, edgecolors = "white", color = color, linewidths = 1.75, label = Label)#linewidths = 1.5
        legend = subplot.legend(loc='lower right', fontsize=12, frameon=True, borderpad=0.8)
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("darkgray")
        all_colors.append(color)
    return all_colors

def get_row_column(sub_fig_num):
    max_col_num = 1
    for i in range(1, 10):
        if i*max_col_num >= sub_fig_num:
            row = i
            break
    for i in range(1, 10):
        if i*row >= sub_fig_num:
            col=i
            break
    return row, col

# Function to plot multiple subgraphs
def plot_scatter_multiple_main(*args):
    sub_fig_num = len(args)
    row, col = get_row_column(sub_fig_num)
    fig, axs = plt.subplots(
        row, 
        col, 
        figsize=(10*col, 8*row), 
        sharex='col', 
        sharey='row',
        layout="constrained"
    )
    if row == 1:
        axs = [axs]
    if col == 1:
        for i in range(len(axs)):
            axs[i] = [axs[i]]
    for r in range(row):
        # Hide Y labels
        for ax in axs[r][1:]:
            plt.setp(ax.get_yticklabels(), visible=False)
    for i in range(sub_fig_num):
        plot_args = [axs[i//col][i%col]]+args[i]
        all_color = plot_scatter_main(*plot_args)
    del all_color[0]
    # all_pos = [
    #     [0.76, 0.75],
    #     [0.18, 0.27],
    #     [0.16, 0.87],
    # ]
    # for color, name, pos in zip(all_color, [" Ambiguous ", " Difficult ", " Easy "], all_pos):
    #     props = dict(boxstyle='round', facecolor='white', edgecolor = color,alpha=1, linewidth=3)
    #     fig.text(pos[0], pos[1], name, fontsize=24, verticalalignment='center', bbox=props)
    fig.supxlabel("Prompt Uncertainty", fontsize=24)
    fig.supylabel("Prediction Probability", fontsize=24)
    plt.yticks(fontsize=14)
    plt.xticks(fontsize=14)
    # axs[0][0].tick_params(axis='both', which='minor', labelsize=20)
    # fig.savefig("Task-Map.pdf", bbox_inches='tight')
    plt.show()    
    
# Function to plot task Map
    
def plot_task_map_Main(model_dir, split, uncertainty, total_run, rank_var = False, all_cls_gen = "all", plot_uncertainty_list = None, x_log = False):
    if all_cls_gen != "all":
        meta_data_file = "task_metadata.json"
        meta_data = json.load(open(meta_data_file, "r"))
    plot_args_list = []
    for run in tqdm(range(total_run)):
        model_name = f"{split}_{uncertainty}_{run}_"
        uncertainty_file = f"{uncertainty.replace('-High', '').replace('-Low', '')}_predicted_examples.jsonl"
        uncertainty_file_path = os.path.join(model_dir, model_name, uncertainty_file)
        pred_data = [json.loads(line.strip())for line in open(uncertainty_file_path, "r").readlines()]
        if "-bald" in uncertainty:
            MeanDict, VarDict = get_uncertainty_mean_var_bald(pred_data)
        else:
            MeanDict, VarDict = get_uncertainty_mean_var(pred_data)
        CM = "tab10"
        if x_log:
            for task, var in VarDict.items():
                VarDict[task] = var ** (1/2)
        MeanList = []
        VarList = []
        if rank_var:
            v_list = []
            for task, var in VarDict.items():
                v_list.append(var)
            v_list = ss.rankdata(v_list)
            for i, task in enumerate(VarDict.keys()):
                VarDict[task] = v_list[i]
        
        for task, Mean in MeanDict.items():
            if (all_cls_gen =="gen" and meta_data[task+".json"]['DefHasLabel'] == "ExactMentioned"):
                continue
            if (all_cls_gen =="cls" and meta_data[task+".json"]['DefHasLabel'] != "ExactMentioned"):
                continue
            MeanList.append(Mean)
            VarList.append(VarDict[task])
        selected_task_list = []
        all_selected_tasks = set()
        for plot_uncertainty in plot_uncertainty_list:
            selected_tasks = get_selected_tasks(split, plot_uncertainty, run+1)
            if all_cls_gen == "cls":
                selected_tasks = [task for task in selected_tasks if meta_data[task+".json"]['DefHasLabel'] == 'ExactMentioned']
            elif all_cls_gen == "gen":
                selected_tasks = [task for task in selected_tasks if meta_data[task+".json"]['DefHasLabel'] != 'ExactMentioned']
            selected_task_list.append(selected_tasks)
            all_selected_tasks = all_selected_tasks.union(set(selected_tasks))

        
        unselected_task_list = set(VarDict.keys()) - all_selected_tasks
        print(len(VarDict.keys()), len(unselected_task_list))
        selected_task_list = [unselected_task_list] + selected_task_list
        
        all_var_list = []
        all_mean_list = []
        
        for selected_tasks in selected_task_list:
            MeanList_selected = []
            VarList_selected = []
            for task in selected_tasks:
                if task in MeanDict:
                    MeanList_selected.append(MeanDict[task])
                    VarList_selected.append(VarDict[task])
            all_var_list.append(VarList_selected)
            all_mean_list.append(MeanList_selected)
        plot_args_list.append([
            all_var_list, 
            all_mean_list,
            ["All tasks"]+ plot_uncertainty_list,
            model_name,
            CM
        ])
    plot_scatter_multiple_main(*plot_args_list)
    
    