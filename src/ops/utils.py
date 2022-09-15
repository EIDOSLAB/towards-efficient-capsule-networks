import json
import logging
import os
import torch
import pathlib
import glob
import wandb
import random
import torch
import distiller
import numpy as np
import pandas as pd
from copy import deepcopy
from pathlib import Path
from torch.nn.modules.utils import _pair
from math import floor

def conv2d_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """
    Utility function for computing spatial output size of conv2d operation.
    It takes a tuple of (h,w) and returns a tuple of (h,w)

    Source: https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/5
    """

    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    pad = _pair(pad)
    dilation = _pair(dilation)

    h = floor(((h_w[0] + (2 * pad[0]) - (dilation[0] * (kernel_size[0] - 1)) - 1) / stride[0]) + 1)
    w = floor(((h_w[1] + (2 * pad[0]) - (dilation[1] * (kernel_size[1] - 1)) - 1) / stride[1]) + 1)
    return h, w


def save_checkpoint(state, is_best, dir, filename, epoch, keep_checkpoints=False):
    if keep_checkpoints:
        complete_path = dir + "current_epoch_{}.pt".format(epoch)
    elif is_best:
        complete_path = dir + "model_best.pt"
    else:
        complete_path = dir + "current_epoch.pt"

    print("Saving {}".format(complete_path))
    torch.save(state, complete_path)
    wandb.save(complete_path)

def set_logger(log_path):
    """
    Set the logger to log info in terminal and file "log_path".

    Example: logging.info("Starting training...")

    Source: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

    :param log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


class Params():
    """
    Class that loads hyperparameters from a json file.
    Example:

    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params

    Source: https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py

    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


def summary(model, config, model_name="VectorCapsNet"):
    logging.info("=================================================================")
    logging.info("Model architectures: ")
    logging.info(model)

    logging.info("Sizes of parameters: ")
    for name, param in model.named_parameters():
        logging.info("{}: {}".format(name, list(param.size())))
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config.n_params = n_params
    logging.info("=================================================================")

    logging.info("----------------------------------------------------------------")

    if model_name == "VectorCapsNet":
        non_trainable_params_primary_caps = config.batch_size * model.num_primary_units if config.primary_num_routing_iterations != 0 else 0
        non_trainable_params_class_caps = config.batch_size * model.num_primary_units * config.num_classes
        non_trainable_params = non_trainable_params_primary_caps + non_trainable_params_class_caps

        logging.info("Total params: %d " % (n_params + non_trainable_params))
        logging.info("Trainable params: %d " % n_params)
        logging.info("Non-trainable params (coupling coefficients for mini-batch) %d " % non_trainable_params)
    else:
        logging.info("Trainable params: %d " % n_params)
    logging.info("----------------------------------------------------------------")

class DotDict(dict):
    """dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __deepcopy__(self, memo=None):
        return DotDict(deepcopy(dict(self), memo=memo))

def save_args(args, dir):
    dict_args = vars(args)
    with open(dir + "/params.json", "w") as outfile:
        json.dump(args, outfile, indent=4)
    wandb.save(dir + "/params.json")

def get_model_best_path(path, checkpoint="model_best*.pth.tar"):
    for file in glob.glob(os.path.join(path, checkpoint)): 
        return file

def get_local_path(path, it):
    checkpoint = "local_model_best_epoch_*-it{}-*.pth.tar".format(it)
    for file in glob.glob(os.path.join(path, checkpoint)): 
        return file

def formatnumbers(x):
    x = str(x).replace('.', ',')
    return x

def create_experiment_folder(config, run_name):
    return os.path.join(config.experiment_name, str(run_name))

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)

def update_output_runs(output, config, run_id):
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.is_file():
        df = pd.read_csv(output)
    else:
        df = pd.DataFrame(columns=['run_id','model','dataset','freeze','backbone_ratio_remain_flops'])
    
    # if configuration is not present, add the new run   
    if not (df[df.columns[1:]] == [config.model, config.dataset, config.freeze, config.backbone_ratio_remain_flops]).all(1).any():
        row = [run_id, config.model, config.dataset, config.freeze, config.backbone_ratio_remain_flops]
        df.loc[len(df)] = row
    # if configuration is already present, replace the run id value
    else:
        condition = (df["model"] == config.model) & (df["dataset"] == config.dataset) & (df["freeze"] == config.freeze) & (df["backbone_ratio_remain_flops"] == config.backbone_ratio_remain_flops)
        df.loc[condition, "run_id"] = run_id
    df.to_csv(output, index=False)

def get_dummy_input(config, device=None):
    """Generate a representative dummy (random) input for the specified dataset.

    If a device is specified, then the dummay_input is moved to that device.
    """
    batch_size = config.batch_size
    dummy_input = torch.randn(batch_size, config.input_channels, config.input_height, config.input_width)
    dummy_target = torch.randint(0, config.num_classes, (batch_size,)).long()
    dummy_input = dummy_input.to(device)
    dummy_target = dummy_target.to(device)
    return dummy_input, dummy_target

# Source https://github.com/IntelLabs/distiller
def weights_sparsity_summary(model, opt=None):
    try:
        df = distiller.weights_sparsity_summary(
            model.module, return_total_sparsity=True
        )
    except AttributeError:
        df = distiller.weights_sparsity_summary(model, return_total_sparsity=True)
    return df[0]["NNZ (dense)"].sum() // 2


def performance_summary(model, dummy_input, opt=None, prefix=""):
    try:
        df = distiller.model_performance_summary(model.module, dummy_input)
    except AttributeError:
        df = distiller.model_performance_summary(model, dummy_input)
    new_entry = {
        "Name": ["Total"],
        "MACs": [df["MACs"].sum()],
    }
    MAC_total = df["MACs"].sum()
    return MAC_total


def model_summary(model, dummy_input, opt=None):
    return (
        performance_summary(model, dummy_input, opt),
        weights_sparsity_summary(model, opt),
    )

