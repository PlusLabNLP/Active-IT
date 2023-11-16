# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Natural Instruction V2 Dataset."""


import json
import os
import random
import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """
@article{wang2022benchmarking,
  title={Benchmarking Generalization via In-Context Instructions on 1,600+ Language Tasks},
  author={Wang, Yizhong and Mishra, Swaroop and Alipoormolabashi, Pegah and Kordi, Yeganeh and others},
  journal={arXiv preprint arXiv:2204.07705},
  year={2022}
}
"""

_DESCRIPTION = """
Natural-Instructions v2 is a benchmark of 1,600+ diverse language tasks and their expert-written instructions. 
It covers 70+ distinct task types, such as tagging, in-filling, and rewriting. 
These tasks are collected with contributions of NLP practitioners in the community and 
through an iterative peer review process to ensure their quality. 
"""

def random_drop_word(text, drop_rate, mask = False):
    text_split = text.split()
    if mask:
        mask_id = 0
        new_text_split = []
        for t in text_split:
            if random.uniform(0, 1) > drop_rate:
                new_text_split.append(t)
            else: # Mask
                mask_token = f"<extra_id_{mask_id}>"
                new_text_split.append(mask_token)
                mask_id += 1
    else:
        new_text_split = [t for t in text_split if random.uniform(0, 1) > drop_rate]
    return " ".join(new_text_split)

def get_id_to_label_dict(pred_file):
    id_to_label_dict = {}
    with open(pred_file, "r") as F:
        for line in F.readlines():
            pred_instance = json.loads(line)
            id_to_label_dict[pred_instance['id']] = pred_instance['prediction']
    return id_to_label_dict

_URL = "https://instructions.apps.allenai.org/"

class NIConfig(datasets.BuilderConfig):
    def __init__(
            self, 
            *args, 
            task_dir=None, 
            max_num_instances_per_task=None, 
            max_num_instances_per_eval_task=None, 
            max_num_instances_per_test_task=None, 
            seed = 42, 
            perturb_inst = None, 
            perturb_num = 0,
            pred_remain = False,
            perturb_def_drop_rate = 0.1,
            perturb_def_mask = False,
            used_label_file = None,
            **kwargs
        ):
        super().__init__(*args, **kwargs)
        self.task_dir: str = task_dir
        self.max_num_instances_per_task: int = max_num_instances_per_task
        self.max_num_instances_per_eval_task: int = max_num_instances_per_eval_task
        self.max_num_instances_per_test_task: int = max_num_instances_per_test_task
        self.seed: int = seed
        self.perturb_inst: str = perturb_inst
        self.perturb_num: int = perturb_num
        self.pred_remain: bool = pred_remain
        self.perturb_def_drop_rate: float = perturb_def_drop_rate
        self.perturb_def_mask: bool = perturb_def_mask
        self.used_label_file: str = used_label_file


class NaturalInstructions(datasets.GeneratorBasedBuilder):
    """NaturalInstructions Dataset."""

    VERSION = datasets.Version("2.0.0")
    BUILDER_CONFIG_CLASS = NIConfig
    BUILDER_CONFIGS = [
        NIConfig(name="default", description="Default config for NaturalInstructions")
    ]
    DEFAULT_CONFIG_NAME = "default"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "Task": datasets.Value("string"),
                    "Contributors": datasets.Value("string"),
                    "Source": [datasets.Value("string")],
                    "URL": [datasets.Value("string")],
                    "Categories": [datasets.Value("string")],
                    "Reasoning": [datasets.Value("string")],
                    "Definition": [datasets.Value("string")],
                    "Positive Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Negative Examples": [{
                        "input": datasets.Value("string"),
                        "output": datasets.Value("string"),
                        "explanation": datasets.Value("string")
                    }],
                    "Input_language": [datasets.Value("string")],
                    "Output_language": [datasets.Value("string")],
                    "Instruction_language": [datasets.Value("string")],
                    "Domains": [datasets.Value("string")],
                    # "Instances": [{
                    #     "input": datasets.Value("string"),
                    #     "output": [datasets.Value("string")]
                    # }],
                    "Instance": {
                        "id": datasets.Value("string"),
                        "input": datasets.Value("string"),
                        "output": [datasets.Value("string")]
                    },
                    "Instance License": [datasets.Value("string")],
                    "Split": datasets.Value("string"),
                    "CLS_GEN": datasets.Value("string")
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/allenai/natural-instructions",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        if self.config.data_dir is None or self.config.task_dir is None:
            dl_path = dl_manager.download_and_extract(_URL)
            self.config.data_dir = self.config.data_dir or os.path.join(dl_path, "splits")
            self.config.task_dir = self.config.task_dir or os.path.join(dl_path, "tasks")

        split_dir = self.config.data_dir
        task_dir = self.config.task_dir

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "path": os.path.join(split_dir, "train_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_task,
                    "subset": "train"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "path": os.path.join(split_dir, "dev_tasks.txt"), 
                    "task_dir": task_dir,
                    "max_num_instances_per_task": self.config.max_num_instances_per_eval_task,
                    "subset": "dev"
                }),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "path": os.path.join(split_dir, "remain_tasks.txt") if self.config.pred_remain else os.path.join(split_dir, "test_tasks.txt"), 
                    "task_dir": task_dir, 
                    "max_num_instances_per_task": self.config.max_num_instances_per_test_task,
                    "subset": "test"
                }),
        ]
    # Send signals to specifiy whether to extend potivie examples for training
    def _generate_examples(self, path=None, task_dir=None, max_num_instances_per_task=None, subset=None):
        """Yields examples."""
        logger.info(f"Generating tasks from = {path}")
        random.seed(self.config.seed)
        print(f"Set random seed in load dataset = {self.config.seed}")
        if self.config.used_label_file is not None:
            print(f"--------------- Use label file: {self.config.used_label_file}")
            instance_id_to_label_dict = get_id_to_label_dict(self.config.used_label_file)
        with open(path, encoding="utf-8") as split_f:
            for line in split_f:
                task_name = line.strip()
                task_path = os.path.join(task_dir, task_name + ".json")
                with open(task_path, encoding="utf-8") as task_f:
                    s = task_f.read()
                    task_data = json.loads(s)
                    task_data["Task"] = task_name
                    if "Instruction Source" in task_data:
                        task_data.pop("Instruction Source")
                    all_instances = task_data.pop("Instances")
                    if subset == "test":
                        # for testing tasks, 100 instances are selected for efficient evaluation and they are label-balanced.
                        # we put them in the first for reproducibility.
                        # so, we use them here
                        if self.config.pred_remain: # Shuffle the instance if predicting remained tasks
                            random.shuffle(all_instances)
                            if self.config.used_label_file is not None:
                                # Make sure the sampled instances have the labels in used_label_file,
                                # which means it should be in instance_id_to_label_dict
                                original_len = len(all_instances)
                                instances_with_new_label = [instance for instance in all_instances if instance['id'] in instance_id_to_label_dict]
                                instances_without_new_label = [instance for instance in all_instances if not instance['id'] in instance_id_to_label_dict]
                                all_instances = instances_with_new_label + instances_without_new_label # Re arrange the list. Put all instances with label at the front
                                assert len(all_instances) == original_len, f"Error! all_instances len went wrong. Should be eqaul to {original_len} but instead get {len(all_instances)}..."
                        instances = all_instances[:max_num_instances_per_task]
                    else:
                        instances = all_instances
                    if max_num_instances_per_task is not None and max_num_instances_per_task >= 0 and subset!="test":
                        random.shuffle(instances)
                        instances = instances[:max_num_instances_per_task]
                    ADD_POS_EXAMPLE_NUM = 20
                    ADD_POS = False
                    if self.config.perturb_inst is not None and subset=="test":
                        ADD_POS = True
                    if ADD_POS:
                        for unused_instance in all_instances[max_num_instances_per_task:][:ADD_POS_EXAMPLE_NUM]: # Seems like too many examples here will cause error
                            task_data["Positive Examples"].append(
                                {
                                    "input": unused_instance['input'],
                                    "output": unused_instance['output'][0],
                                    "explanation" : "None",
                                }
                            )
                            assert type(unused_instance['input']) == str, "input error"
                            assert type(unused_instance['output'][0]) == str, "output error"
                    # For TLAL
                    # Implement the perturbation of instructions
                    # Noted that for k perturb_num, we want to fix it to K type of instructions. Thus we create the perturb instructions beforehand
                    perturb_task_data_list = []
                    for pid in range(self.config.perturb_num):
                        perturb_task_data = task_data.copy()
                        if self.config.perturb_inst in ["TE", "ALL"]:
                            perturb_task_data["Positive Examples"] = random.sample(task_data["Positive Examples"], 2)
                        elif self.config.perturb_inst in ["TD", "ALL", "TDTE"]:
                            # Be aware of the copy. When update deep level element, will change the original one
                            # So perturb_task_data["Definition"][0]= random_drop_word(..) will fail
                            # if pid == 0, don't perturb it. Use the original task definition.
                            if pid == 0:
                                perturb_task_data["Definition"]= [task_data["Definition"][0]]
                            else:
                                if self.config.perturb_def_drop_rate < 1: #If >1, apply some combined dropping
                                    perturb_def_drop_rate = self.config.perturb_def_drop_rate
                                elif self.config.perturb_def_drop_rate == 1:
                                    # Combined for 0.05, to 0.5
                                    #pid start from 1 here 
                                    perturb_def_drop_rate = ((pid-1)//10+1)*0.1
                                elif self.config.perturb_def_drop_rate == 2:
                                    # Combined for 0.1, to 1
                                    #pid start from 1 here
                                    # perturb_def_drop_rate = float(pid)/float(self.config.perturb_num-1)
                                    perturb_def_drop_rate = 0.1
                                elif self.config.perturb_def_drop_rate == 3:
                                    perturb_def_drop_rate = 0.2
                                else:
                                    assert False, "Error here"
                                perturb_task_data["Definition"]= [random_drop_word(
                                    text = task_data["Definition"][0],
                                    drop_rate = perturb_def_drop_rate,
                                    mask = self.config.perturb_def_mask,
                                )]
                                if self.config.perturb_inst == "TDTE":
                                    if (self.config.perturb_def_drop_rate != 3) or (task_data['CLS_GEN'] == "GEN"): #Only drop examples for gen tasks
                                        # Also perturb TE
                                        perturb_task_data["Positive Examples"] = [
                                            # Positive Example #0
                                            {
                                                "input": random_drop_word(
                                                    text = perturb_task_data["Positive Examples"][0]['input'],
                                                    drop_rate = perturb_def_drop_rate,
                                                    mask = self.config.perturb_def_mask,
                                                ),
                                                "output": perturb_task_data["Positive Examples"][0]['output'],
                                                "explanation": perturb_task_data["Positive Examples"][0]['explanation'],
                                            },
                                            # Positive Example #1
                                            {
                                                "input": random_drop_word(
                                                    text = perturb_task_data["Positive Examples"][1]['input'],
                                                    drop_rate = perturb_def_drop_rate,
                                                    mask = self.config.perturb_def_mask,
                                                ),
                                                "output": perturb_task_data["Positive Examples"][1]['output'],
                                                "explanation": perturb_task_data["Positive Examples"][1]['explanation'],
                                            },
                                        ]
                        elif self.config.perturb_inst == "MC": # don't perturb. simply get the loss. MC Dropout
                            pass
                        elif self.config.perturb_inst == "None": # don't perturb. simply get the loss
                            pass
                        else:
                            print(f"Error: perturb_inst type {self.config.perturb_inst} is undefined")
                            exit(0)
                        perturb_task_data_list.append(perturb_task_data)
                    new_instances = []
                    # If specified used_label_file, change the label of instances using instance_id_to_label_dict
                    if self.config.used_label_file is not None:
                        for instance in instances:
                            instance_id = instance['id']
                            assert instance_id in instance_id_to_label_dict, f"Error! instance_id {instance_id} not in instance_id_to_label_dict!!"
                            new_label = instance_id_to_label_dict[instance_id]
                            instance['output'] = [new_label]
                    if self.config.perturb_inst is not None:
                        for idx, instance in enumerate(instances):
                            for pid in range(self.config.perturb_num):
                                example = perturb_task_data_list[pid].copy()
                                example["id"] = instance["id"] + f"-{pid}"
                                example["Instance"] = instance
                                yield f"{task_name}_{idx}_{pid}", example
                    else:
                        for idx, instance in enumerate(instances):
                            example = task_data.copy()
                            example["id"] = instance["id"]
                            example["Instance"] = instance
                            yield f"{task_name}_{idx}", example

