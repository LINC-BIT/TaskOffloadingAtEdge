import logging
import thop
import torch
import torch.nn as nn
import yaml
import os
from branchynet.util import getEarlyExitChain, getAccuracyByEarlyExitPoint
from experiments.image_classification.resnet.resnet_utils import resnet18_branchynet_cifar
from data.image_classification.cifar100 import CIFAR100Dataloader
from tqdm import tqdm
import time

# 假设每个含参数层仅运行一次
class ModelProfiler:
    def __init__(self, model, input_shape, exit_points, save_path, dataloader, device='cpu'):
        self.model = model
        self.exit_points = exit_points
        self.save_path = save_path
        self.input_shape = input_shape
        self.device = device
        self.dataloader = dataloader

    def profile_model(self):
        if not os.path.exists(os.path.join(self.save_path, 'profiles.yaml')):
            self.model.eval()
            hooks = []
            def getHookFn(di, id):
                def hook_fn(module, input, output):
                    tmp = {}
                    if len(input) > 1:
                        tmp['input'] = [list(i.shape[1:]) for i in input]
                    else:
                        tmp['input'] = list(input[0].shape[1:])

                    if isinstance(output, tuple):
                        tmp['output'] = [list(o.shape[1:]) for o in output]
                    else:
                        tmp['output'] = list(output.shape[1:])

                    hooks[id].remove()
                    if len(list(module.children())) == 0:
                        module = nn.Sequential(module)
                    ops, param = thop.profile(module, inputs=input, verbose=False)

                    tmp['FLOPs'] = ops * 2
                    tmp['param'] = param
                    di[f'layer-{id}'] = tmp
                return hook_fn
            profile_dict = {}
            for i, exit_point in enumerate(self.exit_points):
                pbar = tqdm(total=len(self.dataloader))
                pbar.set_description(f'Testing branch-{i+1}')
                tmp = {}
                tmp['acc'] = getAccuracyByEarlyExitPoint(self.model, self.dataloader, exit_point, self.device, pbar=pbar)
                model = getEarlyExitChain(self.model, exit_point)
                tmp['exit point'] = exit_point
                tmp['chain length'] = len(model)

                for id, layer in enumerate(model):
                    hooks.append(layer.register_forward_hook(getHookFn(tmp, id)))

                model(torch.rand(self.input_shape))

                profile_dict[f'Branch-{i+1}'] = tmp
                hooks.clear()

            f = open(os.path.join(self.save_path, 'profiles.yaml'), 'w')
            yaml.dump(profile_dict, f)

def getPartitionedModels(model, s):
    edge_model = []
    server_model = []

    for id, layer in enumerate(model):
        if id + 1 <= s:
            edge_model.append(layer)
        else:
            server_model.append(layer)
    edge_model = nn.Sequential(*edge_model)
    server_model = nn.Sequential(*server_model)
    server_model.eval()
    edge_model.eval()
    return edge_model, server_model

def InferenceOfGPU(model, x, repeat=1):
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ti = 0.
    with torch.no_grad():
        for i in range(repeat):
            starter.record()
            res = model(x)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            ti += curr_time
    ti /= repeat
    return  res, ti

def InferenceOfCPU(model, x, repeat=1):
    ti = 0.
    with torch.no_grad():
        for i in range(repeat):
            sta = time.perf_counter()
            try:
                res = model(x)
            except Exception as e:
                print(e)
            ti += (time.perf_counter() - sta) * 1000
    ti /= repeat
    return res, ti

def Inference(model, x, is_gpu, repeat=1):
    model.eval()
    if is_gpu:
        return InferenceOfGPU(model, x, repeat)
    else:
        return InferenceOfCPU(model, x, repeat)

def warmUP(model, x, repeat=100):
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(x)

if __name__ == '__main__':
    model, points = resnet18_branchynet_cifar(100)
    model_path = '/data/zcr/DDPG_based_DEC_DNN_IO/results/image_classification/resnet/cifar100/resnet.pth'
    state_dict = torch.load(model_path, map_location='cuda')
    model.load_state_dict(state_dict)
    train_dataloader, test_dataloader = CIFAR100Dataloader()
    p = ModelProfiler(model, (1, 3, 32, 32), points, './results/image_classification/resnet/cifar100', test_dataloader)

    p.profile_model()