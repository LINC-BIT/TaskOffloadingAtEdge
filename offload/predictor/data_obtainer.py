import pandas as pd
from offload.predictor.utils import binary_search, getFlopsOfLinear, getFlopsOfConv2d, getFlopsOfDwConv2d
from offload.inference_utils import getGPULatency, getCPULatency, warmUP
import torch.nn as nn
import torch
from multiprocessing.pool import ThreadPool as Pool
import random
import os
import math


def find_datasets(datapath, layer, device, is_gpu, repeat=30):
    path = os.path.join(datapath, f'{device}')
    addr = "gpu" if is_gpu else "cpu"
    num_thread = 1
    if isinstance(layer, nn.Conv2d):
        if layer.groups == 1:
            path = os.path.join(path, f'conv_{addr}.csv')
            df = find_datasets_from_conv(is_gpu, repeat, num_threads=num_thread)
        else:
            path = os.path.join(path, f'dwconv_{addr}.csv')
            df = find_datasets_from_dwconv(is_gpu, repeat, num_threads=num_thread)
    elif isinstance(layer, nn.Linear):
        path = os.path.join(path, f'linear_{addr}.csv')
        df = find_datasets_from_linear(is_gpu, repeat, num_threads=num_thread)
    elif isinstance(layer, nn.MaxPool2d):
        path = os.path.join(path, f'maxpool_{addr}.csv')
        df = find_datasets_from_maxpool(is_gpu, repeat, num_threads=num_thread)
    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        path = os.path.join(path, f'avgpool_{addr}.csv')
        df = find_datasets_from_avgpool(is_gpu, repeat, num_threads=num_thread)
    elif isinstance(layer, nn.BatchNorm2d):
        path = os.path.join(path, f'batchnorm_{addr}.csv')
        df = find_datasets_from_batchnorm(is_gpu, repeat, num_threads=num_thread)
    else:
        raise TypeError('目前不支持所输入层的数据集收集')
    df.to_csv(path, index=False)
    return df

def find_datasets_from_conv(is_gpu, repeat, num_threads=8):
    def calcLatency(process_list):
        for i in process_list:
            t_model = tmodels[binary_search(arrs, i)]
            _cin = ta_dict['cin'][i]
            _hw = ta_dict['HW'][i]
            x = torch.rand(size=(1, _cin, _hw, _hw)).to('cuda' if is_gpu else 'cpu')
            if is_gpu:
                ti = getGPULatency(t_model, x, repeat, starter, ender)
            else:
                ti = getCPULatency(t_model, x, repeat)
            ta_dict['latency'][i] = ti
            print(i)

    if is_gpu:
        assert torch.cuda.is_available(), '该设备不能使用GPU'
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    print('收集卷积的数据中...')
    ta_dict = {'HW': [], 'kernel': [], 'stride': [], 'cin': [], 'cout': [], 'FLOPs': []}
    tmodels = []
    arrs = []
    hw = [1, 7, 8, 13, 14, 27, 28, 32, 56, 112, 224]
    k = [1, 3, 5, 7, 9, 11]
    st = [1, 2, 4]
    cin = [3, 32, 64, 128, 256, 512, 1024]
    cout = [3, 32, 64, 128, 256, 512, 1024]
    task_num = 0

    for _k in k:
        for _st in st:
            for _cin in cin:
                for _cout in cout:
                    arrs.append(task_num)
                    tmodels.append(nn.Conv2d(in_channels=_cin, out_channels=_cout, stride=_st, kernel_size=_k).to('cuda' if is_gpu else 'cpu'))
                    for _hw in hw:
                        if _k > _hw:
                            continue
                        flops = getFlopsOfConv2d(_cin, _cout, _k, 0, _st, _hw)
                        task_num += 1
                        ta_dict['HW'].append(_hw)
                        ta_dict['kernel'].append(_k)
                        ta_dict['stride'].append(_st)
                        ta_dict['cin'].append(_cin)
                        ta_dict['cout'].append(_cout)
                        ta_dict['FLOPs'].append(flops)
    arrs.append(task_num)
    ta_dict['latency'] = [0. for _ in range(task_num)]
    p_list = [i for i in range(task_num)]

    dummy_input = torch.rand(size=(1, 3, 224, 224)).to('cuda' if is_gpu else 'cpu')
    tm = tmodels[0]
    warmUP(tm, dummy_input)
    if not is_gpu:
        random.shuffle(p_list)
        len_list = math.ceil(len(p_list) / num_threads)
        p_lists = [p_list[i:i + len_list] for i in range(0, task_num, len_list)]
        pool = Pool(num_threads)
        pool.map(calcLatency, p_lists)
    else:
        calcLatency(p_list)

    df = pd.DataFrame(ta_dict)
    df['latency'] = df['latency'].round(3)
    return df

def find_datasets_from_avgpool(is_gpu, repeat, num_threads=8):
    def calcLatency(process_list):
        for i in process_list:
            t_model = tmodels[binary_search(arrs, i)]
            _cin = ta_dict['cin'][i]
            _hw = ta_dict['HW'][i]
            x = torch.rand(size=(1, _cin, _hw, _hw)).to('cuda' if is_gpu else 'cpu')
            if is_gpu:
                ti = getGPULatency(t_model, x, repeat, starter, ender)
            else:
                ti = getCPULatency(t_model, x, repeat)
            ta_dict['latency'][i] = ti
            print(i)

    if is_gpu:
        assert torch.cuda.is_available(), '该设备不能使用GPU'
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    print('收集平均池化的数据中...')
    ta_dict = {'HW': [], 'output': [], 'cin': []}
    tmodels = []
    arrs = []
    hw = [1, 7, 8, 13, 14, 27, 28, 32, 56, 112, 224]
    cin = [i for i in range(3, 1024, 30)]
    output = [56, 32, 28, 27, 14, 13, 8, 7, 6, 1]
    task_num = 0

    for _o in output:
        tmodels.append(nn.AdaptiveAvgPool2d((_o, _o)).to('cuda' if is_gpu else 'cpu'))
        arrs.append(task_num)
        for _hw in hw:
            for _cin in cin:
                if _o > _hw:
                    continue
                task_num += 1
                ta_dict['HW'].append(_hw)
                ta_dict['cin'].append(_cin)
                ta_dict['output'].append(_o)
    arrs.append(task_num)
    ta_dict['latency'] = [0. for _ in range(task_num)]
    p_list = [i for i in range(task_num)]

    dummy_input = torch.rand(size=(1, 3, 224, 224)).to('cuda' if is_gpu else 'cpu')
    tm = tmodels[0]
    warmUP(tm, dummy_input)
    if not is_gpu:
        random.shuffle(p_list)
        len_list = math.ceil(len(p_list) / num_threads)
        p_lists = [p_list[i:i + len_list] for i in range(0, task_num, len_list)]
        pool = Pool(num_threads)
        pool.map(calcLatency, p_lists)
    else:
        calcLatency(p_list)

    df = pd.DataFrame(ta_dict)
    df['latency'] = df['latency'].round(3)
    return df

def find_datasets_from_maxpool(is_gpu, repeat, num_threads=8):
    def calcLatency(process_list):
        for i in process_list:
            t_model = tmodels[binary_search(arrs, i)]
            _cin = ta_dict['cin'][i]
            _hw = ta_dict['HW'][i]
            x = torch.rand(size=(1, _cin, _hw, _hw)).to('cuda' if is_gpu else 'cpu')
            if is_gpu:
                ti = getGPULatency(t_model, x, repeat, starter, ender)
            else:
                ti = getCPULatency(t_model, x, repeat)
            ta_dict['latency'][i] = ti
            print(i)

    if is_gpu:
        assert torch.cuda.is_available(), '该设备不能使用GPU'
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    print('收集最大池化的数据中...')
    ta_dict = {'HW': [], 'kernel': [], 'stride': [], 'cin': []}
    tmodels = []
    arrs = []
    hw = [1, 7, 8, 13, 14, 27, 28, 32, 56, 112, 224]
    k = [1, 3, 5, 7, 9, 11]
    st = [1, 2, 4]
    cin = [i for i in range(3, 1024, 50)]
    task_num = 0

    for _k in k:
        for _st in st:
            arrs.append(task_num)
            tmodels.append(nn.MaxPool2d(kernel_size=_k, stride=_st).to('cuda' if is_gpu else 'cpu'))
            for _cin in cin:
                for _hw in hw:
                    if _k > _hw:
                        continue
                    task_num += 1
                    ta_dict['HW'].append(_hw)
                    ta_dict['kernel'].append(_k)
                    ta_dict['stride'].append(_st)
                    ta_dict['cin'].append(_cin)
    arrs.append(task_num)
    ta_dict['latency'] = [0. for _ in range(task_num)]
    p_list = [i for i in range(task_num)]

    dummy_input = torch.rand(size=(1, 3, 224, 224)).to('cuda' if is_gpu else 'cpu')
    tm = tmodels[0]
    warmUP(tm, dummy_input)
    if not is_gpu:
        random.shuffle(p_list)
        len_list = math.ceil(len(p_list) / num_threads)
        p_lists = [p_list[i:i + len_list] for i in range(0, task_num, len_list)]
        pool = Pool(num_threads)
        pool.map(calcLatency, p_lists)
    else:
        calcLatency(p_list)

    df = pd.DataFrame(ta_dict)
    df['latency'] = df['latency'].round(3)
    return df

def find_datasets_from_linear(is_gpu, repeat, num_threads=8):
    def calcLatency(process_list):
        for i in process_list:
            _i = ta_dict['input'][i]
            _o = ta_dict['output'][i]
            t_model = nn.Linear(_i, _o).to('cuda' if is_gpu else 'cpu')
            x = torch.rand((1, _i)).to('cuda' if is_gpu else 'cpu')
            if is_gpu:
                ti = getGPULatency(t_model, x, repeat, starter, ender)
            else:
                ti = getCPULatency(t_model, x, repeat)
            ta_dict['latency'][i] = ti
            print(i)

    if is_gpu:
        assert torch.cuda.is_available(), '该设备不能使用GPU'
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    print('收集线性层的数据中...')
    ta_dict = {'input': [], 'output': [], 'FLOPs': []}
    inputs = [i for i in range(10, 8120, 150)]
    outputs = [i for i in range(10, 8120, 150)]
    task_num = 0

    for _i in inputs:
        for _o in outputs:
            flops = getFlopsOfLinear(_i, _o)
            task_num += 1
            ta_dict['FLOPs'].append(flops)
            ta_dict['input'].append(_i)
            ta_dict['output'].append(_o)
    ta_dict['latency'] = [0. for _ in range(task_num)]
    p_list = [i for i in range(task_num)]

    dummy_input = torch.rand(size=(1, 3)).to('cuda' if is_gpu else 'cpu')
    tm = nn.Linear(3, 3).to('cuda' if is_gpu else 'cpu')
    warmUP(tm, dummy_input)
    if not is_gpu:
        random.shuffle(p_list)
        len_list = math.ceil(len(p_list) / num_threads)
        p_lists = [p_list[i:i + len_list] for i in range(0, task_num, len_list)]
        pool = Pool(num_threads)
        pool.map(calcLatency, p_lists)
    else:
        calcLatency(p_list)

    df = pd.DataFrame(ta_dict)
    df['latency'] = df['latency'].round(3)
    return df

def find_datasets_from_batchnorm(is_gpu, repeat, num_threads=8):
    def calcLatency(process_list):
        for i in process_list:
            t_model = tmodels[i // len_hw]
            _cin = ta_dict['cin'][i]
            _hw = ta_dict['HW'][i]
            x = torch.rand(size=(1, _cin, _hw, _hw)).to('cuda' if is_gpu else 'cpu')
            if is_gpu:
                ti = getGPULatency(t_model, x, repeat, starter, ender)
            else:
                ti = getCPULatency(t_model, x, repeat)
            ta_dict['latency'][i] = ti
            print(i)

    if is_gpu:
        assert torch.cuda.is_available(), '该设备不能使用GPU'
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    print('收集BN的数据中...')
    ta_dict = {'HW': [], 'cin': []}
    tmodels = []
    hw = [7, 8, 13, 14, 27, 28, 32, 56, 112]
    cin = [i for i in range(3, 1024, 5)]
    task_num = 0
    len_hw = len(hw)

    for _cin in cin:
        tmodels.append(nn.BatchNorm2d(_cin).to('cuda' if is_gpu else 'cpu'))
        for _hw in hw:
            task_num += 1
            ta_dict['HW'].append(_hw)
            ta_dict['cin'].append(_cin)
    ta_dict['latency'] = [0. for _ in range(task_num)]
    p_list = [i for i in range(task_num)]

    dummy_input = torch.rand(size=(1, 3, 224, 224)).to('cuda' if is_gpu else 'cpu')
    tm = tmodels[0]
    warmUP(tm, dummy_input)
    if not is_gpu:
        random.shuffle(p_list)
        len_list = math.ceil(len(p_list) / num_threads)
        p_lists = [p_list[i:i + len_list] for i in range(0, task_num, len_list)]
        pool = Pool(num_threads)
        pool.map(calcLatency, p_lists)
    else:
        calcLatency(p_list)

    df = pd.DataFrame(ta_dict)
    df['latency'] = df['latency'].round(3)
    return df

def find_datasets_from_dwconv(is_gpu, repeat, num_threads=8):
    def calcLatency(process_list):
        for i in process_list:
            t_model = tmodels[binary_search(arrs, i)]
            _cin = ta_dict['cin'][i]
            _hw = ta_dict['HW'][i]
            x = torch.rand(size=(1, _cin, _hw, _hw)).to('cuda' if is_gpu else 'cpu')
            if is_gpu:
                ti = getGPULatency(t_model, x, repeat, starter, ender)
            else:
                ti = getCPULatency(t_model, x, repeat)
            ta_dict['latency'][i] = ti
            print(i)

    if is_gpu:
        assert torch.cuda.is_available(), '该设备不能使用GPU'
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    print('收集Depthwise卷积的数据中...')
    ta_dict = {'HW': [], 'kernel': [], 'stride': [], 'cin': [], 'FLOPs': []}
    tmodels = []
    arrs = []
    hw = [1, 7, 8, 13, 14, 27, 28, 32, 56, 112, 224]
    k = [1, 3, 5, 7, 9, 11]
    st = [1, 2, 4]
    cin = [3, 32, 64, 128, 256, 512, 1024]
    task_num = 0

    for _k in k:
        for _st in st:
            for _cin in cin:
                arrs.append(task_num)
                tmodels.append(nn.Conv2d(in_channels=_cin, out_channels=_cin, stride=_st, kernel_size=_k, groups=_cin).to('cuda' if is_gpu else 'cpu'))
                for _hw in hw:
                    if _k > _hw:
                        continue
                    flops = getFlopsOfDwConv2d(_cin, _k, 0, _st, _hw)
                    task_num += 1
                    ta_dict['HW'].append(_hw)
                    ta_dict['kernel'].append(_k)
                    ta_dict['stride'].append(_st)
                    ta_dict['cin'].append(_cin)
                    ta_dict['FLOPs'].append(flops)
    arrs.append(task_num)
    ta_dict['latency'] = [0. for _ in range(task_num)]
    p_list = [i for i in range(task_num)]

    dummy_input = torch.rand(size=(1, 3, 224, 224)).to('cuda' if is_gpu else 'cpu')
    tm = tmodels[0]
    warmUP(tm, dummy_input)
    if not is_gpu:
        random.shuffle(p_list)
        len_list = math.ceil(len(p_list) / num_threads)
        p_lists = [p_list[i:i + len_list] for i in range(0, task_num, len_list)]
        pool = Pool(num_threads)
        pool.map(calcLatency, p_lists)
    else:
        calcLatency(p_list)

    df = pd.DataFrame(ta_dict)
    df['latency'] = df['latency'].round(3)
    return df

def getDatasets(rootpath, layer, device, is_gpu):
    datapath = os.path.join(rootpath, 'datasets')
    path = os.path.join(datapath, f'{device}')
    addr = "gpu" if is_gpu else "cpu"
    if isinstance(layer, nn.Conv2d):
        if layer.groups == 1:
            path = os.path.join(path, f'conv_{addr}.csv')
        else:
            path = os.path.join(path, f'dwconv_{addr}.csv')
    elif isinstance(layer, nn.Linear):
        path = os.path.join(path, f'linear_{addr}.csv')
    elif isinstance(layer, nn.MaxPool2d):
        path = os.path.join(path, f'maxpool_{addr}.csv')
    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        path = os.path.join(path, f'avgpool_{addr}.csv')
    elif isinstance(layer, nn.BatchNorm2d):
        path = os.path.join(path, f'batchnorm_{addr}.csv')
    else:
        raise TypeError('目前不支持所输入层的数据集收集')
    if not os.path.exists(path):
        if is_gpu:
            find_datasets(datapath, layer, device, is_gpu, repeat=100)
        else:
            find_datasets(datapath, layer, device, is_gpu, repeat=30)

    df = pd.read_csv(path)
    return df

if __name__ == '__main__':
    device = 'cloud'
    is_gpu = True
    rpath = '/data/zcr/legodnn/results/legodnn/image_classification/resnet18_cifar100_0-125/offload/'
    getDatasets(rpath, nn.Conv2d(2, 2, 2), device, is_gpu)
    getDatasets(rpath, nn.Conv2d(2, 2, 2, groups=2), device, is_gpu)
    getDatasets(rpath, nn.Linear(2, 2), device, is_gpu)
    getDatasets(rpath, nn.AdaptiveAvgPool2d((2, 2)), device, is_gpu)
    getDatasets(rpath, nn.MaxPool2d(2, 2), device, is_gpu)
    getDatasets(rpath, nn.BatchNorm2d(2), device, is_gpu)