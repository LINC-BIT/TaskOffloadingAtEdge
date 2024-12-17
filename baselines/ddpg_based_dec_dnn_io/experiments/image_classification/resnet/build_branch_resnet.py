import os

import torch
import torch.nn as nn
import random
import numpy as np

from util.data.util import create_dir
from data.image_classification.cifar100 import CIFAR100Dataloader
from experiments.image_classification.resnet.resnet_utils import resnet18_branchynet_cifar
from branchynet.train import trainClassificationBranchyNet

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    task_name = 'image_classification'
    model_name = 'resnet'
    dataset_name = 'cifar100'
    device = 'cuda:0'
    num_classes = 100
    lr = 0.1
    epoch = 500
    wd = 5e-4
    path = os.path.join('./results', task_name, model_name, dataset_name)
    model_path = os.path.join(path, 'resnet.pth')
    create_dir(path)

    model, exit_points = resnet18_branchynet_cifar(num_classes, device)

    train_dataloader, test_dataloader = CIFAR100Dataloader()

    trainClassificationBranchyNet(model, exit_points, train_dataloader, test_dataloader, model_path, epoch_num=epoch, wd=wd, opt='SGD', lr=lr, device=device)
