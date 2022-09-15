
import argparse
import wandb
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import lines, patches

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "serif",
#     "font.serif": ["Computer Modern Roman"],
#     "font.size": 16,
#     })

def main(args):
    api = wandb.Api()
    f = open('wandb_project.json')
    wand_settings = json.load(f)
    wandb.login()
    wandb_path = "{}/{}".format(wand_settings["entity"], wand_settings["project"])

    df = pd.read_csv(args.wandb_runs)

    run_id_resnet50_cifar10_25_freeze = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "cifar10") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 25)]["run_id"].values[0]
    run_resnet50_cifar10_25_freeze = api.run("{}/{}".format(wandb_path, run_id_resnet50_cifar10_25_freeze))

    run_id_resnet50_cifar10_50_freeze = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "cifar10") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 50)]["run_id"].values[0]
    run_resnet50_cifar10_50_freeze = api.run("{}/{}".format(wandb_path, run_id_resnet50_cifar10_50_freeze))

    run_id_resnet50_cifar10_75_freeze = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "cifar10") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 75)]["run_id"].values[0]
    run_resnet50_cifar10_75_freeze = api.run("{}/{}".format(wandb_path, run_id_resnet50_cifar10_75_freeze))

    run_id_resnet50_cifar10_100_freeze = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "cifar10") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 100)]["run_id"].values[0]
    run_resnet50_cifar10_100_freeze = api.run("{}/{}".format(wandb_path, run_id_resnet50_cifar10_100_freeze))

    run_id_resnet50_cifar10_25_finetuned = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "cifar10") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 25)]["run_id"].values[0]
    run_resnet50_cifar10_25_finetuned = api.run("{}/{}".format(wandb_path, run_id_resnet50_cifar10_25_finetuned))

    run_id_resnet50_cifar10_50_finetuned = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "cifar10") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 50)]["run_id"].values[0]
    run_resnet50_cifar10_50_finetuned = api.run("{}/{}".format(wandb_path, run_id_resnet50_cifar10_50_finetuned))

    run_id_resnet50_cifar10_75_finetuned = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "cifar10") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 75)]["run_id"].values[0]
    run_resnet50_cifar10_75_finetuned = api.run("{}/{}".format(wandb_path, run_id_resnet50_cifar10_75_finetuned))

    run_id_resnet50_cifar10_100_finetuned = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "cifar10") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 100)]["run_id"].values[0]
    run_resnet50_cifar10_100_finetuned = api.run("{}/{}".format(wandb_path, run_id_resnet50_cifar10_100_finetuned))

    run_id_resnet50_tiny_25_freeze = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 25)]["run_id"].values[0]
    run_resnet50_tiny_25_freeze = api.run("{}/{}".format(wandb_path, run_id_resnet50_tiny_25_freeze))

    run_id_resnet50_tiny_50_freeze = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 50)]["run_id"].values[0]
    run_resnet50_tiny_50_freeze = api.run("{}/{}".format(wandb_path, run_id_resnet50_tiny_50_freeze))

    run_id_resnet50_tiny_75_freeze = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 75)]["run_id"].values[0]
    run_resnet50_tiny_75_freeze = api.run("{}/{}".format(wandb_path, run_id_resnet50_tiny_75_freeze))

    run_id_resnet50_tiny_100_freeze = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 100)]["run_id"].values[0]
    run_resnet50_tiny_100_freeze = api.run("{}/{}".format(wandb_path, run_id_resnet50_tiny_100_freeze))

    run_id_resnet50_tiny_25_finetuned = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 25)]["run_id"].values[0]
    run_resnet50_tiny_25_finetuned = api.run("{}/{}".format(wandb_path, run_id_resnet50_tiny_25_finetuned))

    run_id_resnet50_tiny_50_finetuned = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 50)]["run_id"].values[0]
    run_resnet50_tiny_50_finetuned = api.run("{}/{}".format(wandb_path, run_id_resnet50_tiny_50_finetuned))
    
    run_id_resnet50_tiny_75_finetuned = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 75)]["run_id"].values[0]
    run_resnet50_tiny_75_finetuned = api.run("{}/{}".format(wandb_path, run_id_resnet50_tiny_75_finetuned))

    run_id_resnet50_tiny_100_finetuned = df[(df['model'] == "ResNet50VectorCapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 100)]["run_id"].values[0]
    run_resnet50_tiny_100_finetuned = api.run("{}/{}".format(wandb_path, run_id_resnet50_tiny_100_finetuned))

    run_id_mobilenet_cifar10_50_freeze = df[(df['model'] == "Mobilev1CapsNet") & (df['dataset'] == "cifar10") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 50)]["run_id"].values[0]
    run_mobilenet_cifar10_50_freeze = api.run("{}/{}".format(wandb_path, run_id_mobilenet_cifar10_50_freeze))

    run_id_mobilenet_cifar10_100_freeze = df[(df['model'] == "Mobilev1CapsNet") & (df['dataset'] == "cifar10") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 100)]["run_id"].values[0]
    run_mobilenet_cifar10_100_freeze = api.run("{}/{}".format(wandb_path, run_id_mobilenet_cifar10_100_freeze))

    run_id_mobilenet_cifar10_50_finetuned = df[(df['model'] == "Mobilev1CapsNet") & (df['dataset'] == "cifar10") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 50)]["run_id"].values[0]
    run_mobilenet_cifar10_50_finetuned = api.run("{}/{}".format(wandb_path, run_id_mobilenet_cifar10_50_finetuned))

    run_id_mobilenet_cifar10_100_finetuned = df[(df['model'] == "Mobilev1CapsNet") & (df['dataset'] == "cifar10") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 100)]["run_id"].values[0]
    run_mobilenet_cifar10_100_finetuned = api.run("{}/{}".format(wandb_path, run_id_mobilenet_cifar10_100_finetuned))

    run_id_mobilenet_tiny_50_freeze = df[(df['model'] == "Mobilev1CapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 50)]["run_id"].values[0]
    run_mobilenet_tiny_50_freeze = api.run("{}/{}".format(wandb_path, run_id_mobilenet_tiny_50_freeze))

    run_id_mobilenet_tiny_100_freeze = df[(df['model'] == "Mobilev1CapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']) & (df['backbone_ratio_remain_flops'] == 100)]["run_id"].values[0]
    run_mobilenet_tiny_100_freeze = api.run("{}/{}".format(wandb_path, run_id_mobilenet_tiny_100_freeze))

    run_id_mobilenet_tiny_50_finetuned = df[(df['model'] == "Mobilev1CapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 50)]["run_id"].values[0]
    run_mobilenet_tiny_50_finetuned = api.run("{}/{}".format(wandb_path, run_id_mobilenet_tiny_50_finetuned))

    run_id_mobilenet_tiny_100_finetuned = df[(df['model'] == "Mobilev1CapsNet") & (df['dataset'] == "tiny-imagenet-200") & (df['freeze']==False) & (df['backbone_ratio_remain_flops'] == 100)]["run_id"].values[0]
    run_mobilenet_tiny_100_finetuned = api.run("{}/{}".format(wandb_path, run_id_mobilenet_tiny_100_finetuned))

    CB91_Blue = '#2CBDFE'
    CB91_Green = '#47DBCD'
    CB91_Pink = '#F3A0F2'
    CB91_Purple = '#9D2EC5'
    CB91_Violet = '#661D98'
    CB91_Amber = '#F5B14C'

    cols = ["CIFAR10", "Tiny ImageNet"]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    plt.setp(axes.flat, xlabel='Total FLOPS (B)', ylabel='Accuracy')

    #resnet50 cifar10 freeze
    resnet50_cifar10_25_freeze_flops = run_resnet50_cifar10_25_freeze.summary["tot_flops"]
    resnet50_cifar10_25_freeze_acc = run_resnet50_cifar10_25_freeze.summary["best_test_acc"]
    resnet50_cifar10_50_freeze_flops = run_resnet50_cifar10_50_freeze.summary["tot_flops"]
    resnet50_cifar10_50_freeze_acc = run_resnet50_cifar10_50_freeze.summary["best_test_acc"]
    resnet50_cifar10_75_freeze_flops = run_resnet50_cifar10_75_freeze.summary["tot_flops"]
    resnet50_cifar10_75_freeze_acc = run_resnet50_cifar10_75_freeze.summary["best_test_acc"]
    resnet50_cifar10_100_freeze_flops = run_resnet50_cifar10_100_freeze.summary["tot_flops"]
    resnet50_cifar10_100_freeze_acc = run_resnet50_cifar10_100_freeze.summary["best_test_acc"]

    #resnet50 cifar10 finetuned
    resnet50_cifar10_25_finetuned_flops = run_resnet50_cifar10_25_finetuned.summary["tot_flops"]
    resnet50_cifar10_25_finetuned_acc = run_resnet50_cifar10_25_finetuned.summary["best_test_acc"]
    resnet50_cifar10_50_finetuned_flops = run_resnet50_cifar10_50_finetuned.summary["tot_flops"]
    resnet50_cifar10_50_finetuned_acc = run_resnet50_cifar10_50_finetuned.summary["best_test_acc"]
    resnet50_cifar10_75_finetuned_flops = run_resnet50_cifar10_75_finetuned.summary["tot_flops"]
    resnet50_cifar10_75_finetuned_acc = run_resnet50_cifar10_75_finetuned.summary["best_test_acc"]
    resnet50_cifar10_100_finetuned_flops = run_resnet50_cifar10_100_finetuned.summary["tot_flops"]
    resnet50_cifar10_100_finetuned_acc = run_resnet50_cifar10_100_finetuned.summary["best_test_acc"]

    #resnet50 tiny freeze
    resnet50_tiny_25_freeze_flops = run_resnet50_tiny_25_freeze.summary["tot_flops"]
    resnet50_tiny_25_freeze_acc = run_resnet50_tiny_25_freeze.summary["best_test_acc"]
    resnet50_tiny_50_freeze_flops = run_resnet50_tiny_50_freeze.summary["tot_flops"]
    resnet50_tiny_50_freeze_acc = run_resnet50_tiny_50_freeze.summary["best_test_acc"]
    resnet50_tiny_75_freeze_flops = run_resnet50_tiny_75_freeze.summary["tot_flops"]
    resnet50_tiny_75_freeze_acc = run_resnet50_tiny_75_freeze.summary["best_test_acc"]
    resnet50_tiny_100_freeze_flops = run_resnet50_tiny_100_freeze.summary["tot_flops"]
    resnet50_tiny_100_freeze_acc = run_resnet50_tiny_100_freeze.summary["best_test_acc"]

    #resnet50 tiny finetuned
    resnet50_tiny_25_finetuned_flops = run_resnet50_tiny_25_finetuned.summary["tot_flops"]
    resnet50_tiny_25_finetuned_acc = run_resnet50_tiny_25_finetuned.summary["best_test_acc"]
    resnet50_tiny_50_finetuned_flops = run_resnet50_tiny_50_finetuned.summary["tot_flops"]
    resnet50_tiny_50_finetuned_acc = run_resnet50_tiny_50_finetuned.summary["best_test_acc"]
    resnet50_tiny_75_finetuned_flops = run_resnet50_tiny_75_finetuned.summary["tot_flops"]
    resnet50_tiny_75_finetuned_acc = run_resnet50_tiny_75_finetuned.summary["best_test_acc"]
    resnet50_tiny_100_finetuned_flops = run_resnet50_tiny_100_finetuned.summary["tot_flops"]
    resnet50_tiny_100_finetuned_acc = run_resnet50_tiny_100_finetuned.summary["best_test_acc"]

    #mobilenet cifar10 freeze
    mobilenet_cifar10_50_freeze_flops = run_mobilenet_cifar10_50_freeze.summary["tot_flops"]
    mobilenet_cifar10_50_freeze_acc = run_mobilenet_cifar10_50_freeze.summary["best_test_acc"]
    mobilenet_cifar10_100_freeze_flops = run_mobilenet_cifar10_100_freeze.summary["tot_flops"]
    mobilenet_cifar10_100_freeze_acc = run_mobilenet_cifar10_100_freeze.summary["best_test_acc"]

    #mobilenet cifar10 finetuned
    mobilenet_cifar10_50_finetuned_flops = run_mobilenet_cifar10_50_finetuned.summary["tot_flops"]
    mobilenet_cifar10_50_finetuned_acc = run_mobilenet_cifar10_50_finetuned.summary["best_test_acc"]
    mobilenet_cifar10_100_finetuned_flops = run_mobilenet_cifar10_100_finetuned.summary["tot_flops"]
    mobilenet_cifar10_100_finetuned_acc = run_mobilenet_cifar10_100_finetuned.summary["best_test_acc"]

    #mobilenet tiny freeze
    mobilenet_tiny_50_freeze_flops = run_mobilenet_tiny_50_freeze.summary["tot_flops"]
    mobilenet_tiny_50_freeze_acc = run_mobilenet_tiny_50_freeze.summary["best_test_acc"]
    mobilenet_tiny_100_freeze_flops = run_mobilenet_tiny_100_freeze.summary["tot_flops"]
    mobilenet_tiny_100_freeze_acc = run_mobilenet_tiny_50_freeze.summary["best_test_acc"]

    #mobilenet tiny finetuned
    mobilenet_tiny_50_finetuned_flops = run_mobilenet_tiny_50_finetuned.summary["tot_flops"]
    mobilenet_tiny_50_finetuned_acc = run_mobilenet_tiny_50_finetuned.summary["best_test_acc"]
    mobilenet_tiny_100_finetuned_flops = run_mobilenet_tiny_100_finetuned.summary["tot_flops"]
    mobilenet_tiny_100_finetuned_acc = run_mobilenet_tiny_100_finetuned.summary["best_test_acc"]

    for i, ax in enumerate(axes):
        ax.annotate(cols[i], xy=(0.5, 1), xytext=(0, 5),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
        ax.set_ylim([0.50, 1])

    CB91_Grad_BP = ['#d14b61', '#2fb9fc', '#33b4fa', '#36b0f8',
                    '#3aacf6', '#3da8f4', '#41a3f2', '#449ff0',
                    '#489bee', '#4b97ec', '#4f92ea', '#528ee8',
                    '#568ae6', '#5986e4', '#5c81e2', '#607de0',
                    '#6379de', '#6775dc', '#6a70da', '#6e6cd8',
                    '#7168d7', '#7564d5', '#785fd3', '#7c5bd1',
                    '#7f57cf', '#8353cd', '#864ecb', '#894ac9',
                    '#8d46c7', '#9042c5', '#943dc3', '#9739c1',
                    '#9b35bf', '#9e31bd', '#a22cbb', '#a528b9',
                    '#a924b7', '#ac20b5', '#b01bb3', '#4b61d1']

    axes[0].scatter([resnet50_cifar10_25_freeze_flops], [resnet50_cifar10_25_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*0.15, label='-25\% flops')
    axes[0].scatter([resnet50_cifar10_50_freeze_flops], [resnet50_cifar10_50_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*0.4, label='-50\% flops')
    axes[0].scatter([resnet50_cifar10_75_freeze_flops], [resnet50_cifar10_75_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*0.75, label='-75\% flops')
    axes[0].scatter([resnet50_cifar10_100_freeze_flops], [resnet50_cifar10_100_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*1.4, label='100\% flops')
    axes[0].plot([resnet50_cifar10_100_freeze_flops, resnet50_cifar10_75_freeze_flops, resnet50_cifar10_50_freeze_flops, resnet50_cifar10_25_freeze_flops], [resnet50_cifar10_100_freeze_acc, resnet50_cifar10_75_freeze_acc, resnet50_cifar10_50_freeze_acc, resnet50_cifar10_25_freeze_acc], c=CB91_Grad_BP[0],linestyle='dashed')
    axes[0].plot([resnet50_cifar10_100_finetuned_flops, resnet50_cifar10_75_finetuned_flops, resnet50_cifar10_50_finetuned_flops, resnet50_cifar10_25_finetuned_flops], [resnet50_cifar10_100_finetuned_acc, resnet50_cifar10_75_finetuned_acc, resnet50_cifar10_50_finetuned_acc, resnet50_cifar10_25_finetuned_acc], c=CB91_Grad_BP[-1],linestyle='dashed')

    axes[0].scatter([resnet50_cifar10_25_finetuned_flops], [resnet50_cifar10_25_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*0.15, label='-25\% flops')
    axes[0].scatter([resnet50_cifar10_50_finetuned_flops], [resnet50_cifar10_50_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*0.4, label='-50\% flops')
    axes[0].scatter([resnet50_cifar10_75_finetuned_flops], [resnet50_cifar10_75_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*0.75, label='-75\% flops')
    axes[0].scatter([resnet50_cifar10_100_finetuned_flops], [resnet50_cifar10_100_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*1.4, label='100\% flops')

    axes[1].scatter([resnet50_tiny_25_freeze_flops], [resnet50_tiny_25_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*0.15,label='-25\% flops')
    axes[1].scatter([resnet50_tiny_50_freeze_flops], [resnet50_tiny_50_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*0.4,label='-50\% flops')
    axes[1].scatter([resnet50_tiny_75_freeze_flops], [resnet50_tiny_75_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*0.75,label='-75\% flops')
    axes[1].scatter([resnet50_tiny_100_freeze_flops], [resnet50_tiny_100_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*1.4,label='100\% flops')
    axes[1].plot([resnet50_tiny_100_freeze_flops, resnet50_tiny_75_freeze_flops, resnet50_tiny_50_freeze_flops, resnet50_tiny_25_freeze_flops], [resnet50_tiny_100_freeze_acc, resnet50_tiny_75_freeze_acc, resnet50_tiny_50_freeze_acc, resnet50_tiny_25_freeze_acc], c=CB91_Grad_BP[0],linestyle='dashed')
    axes[1].plot([resnet50_tiny_100_finetuned_flops, resnet50_tiny_75_finetuned_flops, resnet50_tiny_50_finetuned_flops, resnet50_tiny_25_finetuned_flops], [resnet50_tiny_100_finetuned_acc, resnet50_tiny_75_finetuned_acc, resnet50_tiny_50_finetuned_acc, resnet50_tiny_25_finetuned_acc], c=CB91_Grad_BP[-1],linestyle='dashed')

    axes[1].scatter([resnet50_tiny_25_finetuned_flops], [resnet50_tiny_25_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*0.15,label='-25\% flops')
    axes[1].scatter([resnet50_tiny_50_finetuned_flops], [resnet50_tiny_50_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*0.4,label='-50\% flops')
    axes[1].scatter([resnet50_tiny_75_finetuned_flops], [resnet50_tiny_75_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*0.75,label='-75\% flops')
    axes[1].scatter([resnet50_tiny_100_finetuned_flops], [resnet50_tiny_100_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*1.4, label='100\% flops')

    axes[0].scatter([mobilenet_cifar10_50_freeze_flops], [mobilenet_cifar10_50_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*0.4, label='-50\% flops', linewidth=1)
    axes[0].scatter([mobilenet_cifar10_100_freeze_flops], [mobilenet_cifar10_100_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*1.4, label='100\% flops', linewidth=1)
    axes[0].scatter([mobilenet_cifar10_50_finetuned_flops], [mobilenet_cifar10_50_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*0.4,label='-50\% flops', linewidth=1)
    axes[0].scatter([mobilenet_cifar10_100_finetuned_flops], [mobilenet_cifar10_100_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*1.4,label='100\% flops', linewidth=1)
    axes[0].plot([mobilenet_cifar10_50_freeze_flops, mobilenet_cifar10_100_freeze_flops], [mobilenet_cifar10_50_freeze_acc, mobilenet_cifar10_100_freeze_acc], c=CB91_Grad_BP[0])
    axes[0].plot([mobilenet_cifar10_50_finetuned_flops, mobilenet_cifar10_100_finetuned_flops], [mobilenet_cifar10_50_finetuned_acc, mobilenet_cifar10_100_finetuned_acc],  c=CB91_Grad_BP[-1])

    axes[1].scatter([mobilenet_tiny_50_freeze_flops], [mobilenet_tiny_50_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*0.4, label='-50\% flops', linewidth=1)
    axes[1].scatter([mobilenet_tiny_100_freeze_flops], [mobilenet_tiny_100_freeze_acc], c=CB91_Grad_BP[0], marker='o', s=400*1.4,label='100\% flops', linewidth=1)
    axes[1].scatter([mobilenet_tiny_50_finetuned_flops], [mobilenet_tiny_50_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*0.4,label='-50\% flops', linewidth=1)
    axes[1].scatter([mobilenet_tiny_100_finetuned_flops], [mobilenet_tiny_100_finetuned_acc], c=CB91_Grad_BP[-1], marker='o', s=400*1.4,label='100\% flops', linewidth=1)
    axes[1].plot([mobilenet_tiny_50_freeze_flops, mobilenet_tiny_100_freeze_flops], [mobilenet_tiny_50_freeze_acc, mobilenet_tiny_100_freeze_acc], c=CB91_Grad_BP[0])
    axes[1].plot([mobilenet_tiny_50_finetuned_flops, mobilenet_tiny_100_finetuned_flops], [mobilenet_tiny_50_finetuned_acc, mobilenet_tiny_100_finetuned_acc], c=CB91_Grad_BP[-1])

    freeze_leg = patches.Patch(color=CB91_Grad_BP[0], label='Freezed')
    finetuned_leg = patches.Patch(color=CB91_Grad_BP[-1], label='Finetuned')
    marker_75 = lines.Line2D([], [], color='white', marker='o', markersize=10*1.65, markerfacecolor='#212121', markeredgecolor="#212121", alpha=0.6, label='25\% pruned flops')
    marker_50 = lines.Line2D([], [], color='white', marker='o', markersize=10*1.40, markerfacecolor='#212121', markeredgecolor="#212121",alpha=0.6,label='50\% pruned flops')
    marker_25 = lines.Line2D([], [], color='white', marker='o', markersize=10*1.15, markerfacecolor='#212121', markeredgecolor="#212121",alpha=0.6,label='75\% pruned flops')
    marker_100 = lines.Line2D([], [], color='white', marker='o', markersize=10*1.80, markerfacecolor='#212121', markeredgecolor="#212121",alpha=0.6,label='0\% pruned flops')

    marker_resnet = lines.Line2D([], [], color="#212121", linestyle='--', label='ResNet-50')
    marker_mobilenet = lines.Line2D([], [], color="#212121", linestyle='-', label='MobileNet')

    axes[1].legend(bbox_to_anchor=(0.89, 1.0), loc="upper center", handles=[marker_100, marker_75, marker_50,marker_25, marker_resnet, marker_mobilenet, freeze_leg, finetuned_leg],fontsize=14,ncol=1, markerfirst=False)
    axes[0].set_xscale('log')
    axes[1].set_xscale('log')
    fig.tight_layout()
    fig.subplots_adjust(left=0.08)
    #plt.savefig("icip_flops_vs_acc.pdf")
    plt.show()
    columns = ['Backbone pruned FLOPS (%)','Bottleneck size','Primary caps','Total FLOPS (B)','GPU memory consumption (GB)', 'Training time (epoch, s)', 'Accuracy']
    df = pd.DataFrame(columns=columns)
    row1 = [int(100-run_resnet50_tiny_100_finetuned.config["backbone_ratio_remain_flops"]), 
           int(run_resnet50_tiny_100_finetuned.config["bottleneck_size"]),
           int(run_resnet50_tiny_100_finetuned.config["num_primaryCaps_types"]),
           np.round(run_resnet50_tiny_100_finetuned.summary["tot_flops"],1),
           np.round(np.max(run_resnet50_tiny_100_finetuned.history(keys=["used_gpu"])["used_gpu"]),2),
           int(np.round(np.mean(run_resnet50_tiny_100_finetuned.history(keys=["training_time"])["training_time"]))),
           np.round(np.mean(run_resnet50_tiny_100_finetuned.history(keys=["test/accuracy"])["test/accuracy"]),2)]
    row2 = [int(100-run_resnet50_tiny_75_finetuned.config["backbone_ratio_remain_flops"]), 
        int(run_resnet50_tiny_75_finetuned.config["bottleneck_size"]),
        int(run_resnet50_tiny_75_finetuned.config["num_primaryCaps_types"]),
        np.round(run_resnet50_tiny_75_finetuned.summary["tot_flops"],1),
        np.round(np.max(run_resnet50_tiny_75_finetuned.history(keys=["used_gpu"])["used_gpu"]),2),
        int(np.round(np.mean(run_resnet50_tiny_75_finetuned.history(keys=["training_time"])["training_time"]))),
        np.round(np.mean(run_resnet50_tiny_75_finetuned.history(keys=["test/accuracy"])["test/accuracy"]),2)]
    row3 = [int(100-run_resnet50_tiny_50_finetuned.config["backbone_ratio_remain_flops"]), 
        int(run_resnet50_tiny_50_finetuned.config["bottleneck_size"]),
        int(run_resnet50_tiny_50_finetuned.config["num_primaryCaps_types"]),
        np.round(run_resnet50_tiny_50_finetuned.summary["tot_flops"],1),
        np.round(np.max(run_resnet50_tiny_50_finetuned.history(keys=["used_gpu"])["used_gpu"]),2),
        int(np.round(np.mean(run_resnet50_tiny_50_finetuned.history(keys=["training_time"])["training_time"]))),
        np.round(np.mean(run_resnet50_tiny_50_finetuned.history(keys=["test/accuracy"])["test/accuracy"]),2)]
    row4 = [int(100-run_resnet50_tiny_25_finetuned.config["backbone_ratio_remain_flops"]), 
        int(run_resnet50_tiny_25_finetuned.config["bottleneck_size"]),
        int(run_resnet50_tiny_25_finetuned.config["num_primaryCaps_types"]),
        np.round(run_resnet50_tiny_25_finetuned.summary["tot_flops"],1),
        np.round(np.max(run_resnet50_tiny_25_finetuned.history(keys=["used_gpu"])["used_gpu"]),2),
        int(np.round(np.mean(run_resnet50_tiny_25_finetuned.history(keys=["training_time"])["training_time"]))),
        np.round(np.mean(run_resnet50_tiny_25_finetuned.history(keys=["test/accuracy"])["test/accuracy"]),2)]
    
    df = df.append([pd.Series(row1, index=df.columns[:len(row1)]),
                    pd.Series(row2, index=df.columns[:len(row2)]),
                    pd.Series(row3, index=df.columns[:len(row3)]),
                    pd.Series(row4, index=df.columns[:len(row4)])],ignore_index=True)
    print(df)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Visualize results')
    parser.add_argument('--wandb_runs', default="wandb_runs.csv", type=str, help='runs path')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_arguments()
    main(args)