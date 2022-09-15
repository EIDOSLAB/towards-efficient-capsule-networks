#  Copyright (c) 2022 EIDOSLab. All rights reserved.
#  See the LICENSE file for licensing terms (BSD-style).

"""
PyTorch implementation of Towards Efficient Capsule Networks
Accepted at ICIP 2022 
Special Session SCENA: Simplification, Compression and Efficiency with Neural networks and Artificial intelligence
"""

import os
import torch
import time
import logging
import json
import wandb
import argparse
import loss.capsule_loss as cl
import ops.utils as utils
import torch.nn as nn
from models.mobilev1CapsNet import Mobilev1CapsNet
from models.resNet50CapsNet import ResNet50VectorCapsNet
from ops.utils import save_args
from ops.utils import update_output_runs
from ops.utils import get_dummy_input
from dataloaders.load_data import get_dataloader
from ops.utils import model_summary
from train import train
from test import test
from torch.utils.tensorboard import SummaryWriter
wandb.login()

def train_test_caps(config, output):
    # Opening wandb setting file
    f = open('wandb_project.json')
    wand_settings = json.load(f)
    run = wandb.init(project=wand_settings["project"], entity=wand_settings["entity"], reinit=True)

    update_output_runs(output, config, wandb.run.id)

    experiment_folder = utils.create_experiment_folder(config, wandb.run.id)

    utils.set_seed(config.seed)

    test_base_dir = "results/" + config.dataset + "/" + experiment_folder

    logdir = test_base_dir + "/logs/"
    checkpointsdir = test_base_dir + "/checkpoints/"
    runsdir = test_base_dir + "/runs/"
    imgdir = test_base_dir + "/images/"

    # Make model checkpoint directory
    if not os.path.exists(checkpointsdir):
        os.makedirs(checkpointsdir)

    # Make log directory
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # Make img directory
    if not os.path.exists(imgdir):
        os.makedirs(imgdir)

    # Set logger path
    utils.set_logger(os.path.join(logdir, "model.log"))

    # Get dataset loaders
    train_loader, valid_loader, test_loader = get_dataloader(config, config.data_dir)

    # Enable GPU usage
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device(config.cuda_device)
    else:
        device = torch.device("cpu")

    if config.model == "ResNet50VectorCapsNet":
        caps_model = ResNet50VectorCapsNet(config, device)
    elif config.model == "Mobilev1CapsNet":
        caps_model = Mobilev1CapsNet(config, device)

    config.bottleneck_size = caps_model.c0
    config.num_primaryCaps_types = caps_model.num_primaryCaps_types
    wandb.config.update(config)    
    wandb.watch(caps_model, log="all")

    utils.summary(caps_model, config, config.model)

    caps_criterion = cl.CapsLoss(config.caps_loss,
                                 config.margin_loss_lambda,
                                 config.batch_averaged,
                                 config.m_plus,
                                 config.m_minus,
                                 device)

    if config.optimizer == "adam":
        caps_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, caps_model.parameters()), lr=config.lr)
    else:
        caps_optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, caps_model.parameters()), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    
    if config.scheduler == "exponential":
        caps_scheduler = torch.optim.lr_scheduler.ExponentialLR(caps_optimizer, config.decay_rate)
    elif config.scheduler == "multistep":
        caps_scheduler = torch.optim.lr_scheduler.MultiStepLR(caps_optimizer, milestones=[100, 150], gamma=0.1)
    elif config.scheduler == "constant":
        caps_scheduler = None
    else:
        print('Learning scheduler is not supported!')

    caps_model.to(device)    
    dummy_input, _ = get_dummy_input(config, device)
    flops, params_after = model_summary(caps_model, dummy_input)
    if config.freeze:
        k = 2
    else:
        k = 3
    flops = flops * k / config.batch_size / 1000000000
    wandb.run.summary["tot_flops"] = flops

    print("MODULES")
    for n, m in caps_model.named_modules():
        print(n, m.__class__)

    for state in caps_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    # Print the model architecture and parameters
    utils.summary(caps_model, config, config.model)

    # Save current settings (hyperparameters etc.)
    save_args(config, test_base_dir)

    # Writer for TensorBoard
    writer = None
    if config.tensorboard:
        writer = SummaryWriter(runsdir)

    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    logging.info("Device: {}".format(device))

    if caps_scheduler is not None:
        logging.info("Initial learning rate: {}".format(caps_scheduler.get_last_lr()[0]))
    else:
        logging.info("Initial learning rate: {}".format(config.lr))
    logging.info("Number of routing iterations: {}".format(config.num_routing_iterations))

    best_loss = float('inf')

    epoch = 0
    best_epoch = 0
    training = True
    while training:
        # Start training
        logging.info("Number of routing iterations: {}".format(caps_model.classCaps.num_iterations))
        
        start_training_time = time.time() 
        train(logging, config, train_loader, caps_model, caps_criterion, caps_optimizer, caps_scheduler, writer, epoch, device)
        end_training_time = time.time()
        tot_training_time = end_training_time - start_training_time
        wandb.log({'training_time': tot_training_time}, step=epoch)
        # Start validation
        val_loss, val_acc = test(logging, config, valid_loader, caps_model, caps_criterion, writer, epoch, device,
                         imgdir, split="validation")
        print(val_acc)
        # Start testing
        test_loss, test_acc = test(logging, config, test_loader, caps_model, caps_criterion, writer, epoch, device, imgdir, split="test")

        if writer:
            writer.add_scalar('routing/iterations', caps_model.classCaps.num_iterations, epoch)
            if caps_scheduler is not None:
                writer.add_scalar('lr', caps_scheduler.get_last_lr()[0], epoch)
        
        wandb.log({'routing/iterations': caps_model.classCaps.num_iterations}, step=epoch)

        formatted_epoch = str(epoch).zfill(len(str(config.epochs))+1)
        
        checkpoint_filename = "epoch_{}".format(formatted_epoch)

        if val_loss < best_loss:
            utils.save_checkpoint({
                "epoch": epoch,
                "routing_iterations": caps_model.classCaps.num_iterations,
                "state_dict": caps_model.state_dict(),
                "metric": config.monitor
                #"optimizer": caps_optimizer.state_dict(),
                #"scheduler": caps_scheduler.state_dict(),
            }, True, checkpointsdir, checkpoint_filename, formatted_epoch)
            best_epoch = epoch
            best_loss = val_loss
            wandb.run.summary["best_epoch"] = epoch
            wandb.run.summary["best_test_acc"] = test_acc
            wandb.run.summary["best_test_loss"] = test_loss
            wandb.run.summary["best_val_acc"] = val_acc
            wandb.run.summary["best_val_loss"] = val_loss

        # Save current epoch checkpoint
        utils.save_checkpoint({
            "epoch": epoch,
            "routing_iterations": caps_model.classCaps.num_iterations,
            "state_dict": caps_model.state_dict(),
            "metric": config.monitor
            #"optimizer": caps_optimizer.state_dict(),
            #"scheduler": caps_scheduler.state_dict(),
        }, False, checkpointsdir, checkpoint_filename, formatted_epoch, False)
        epoch += 1

        if epoch - best_epoch > config.patience:
            training = False

    if writer:
        writer.close()
    wandb.save(checkpointsdir+"/*.pt")
    run.finish()

def main(config, output):
    for k in range(len(config.seeds)):
        config.seed = config.seeds[k]
        train_test_caps(config, output)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--config', default=None, type=str, help='Config.json path')
    parser.add_argument('--output', default=None, type=str, help='wandb.json file to track runs')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    config = utils.DotDict(json.load(open(args.config)))
    main(config, args.output)