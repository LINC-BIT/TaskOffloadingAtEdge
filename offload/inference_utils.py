import os
from offload.model import *
import time
import numpy as np
import psutil
import matplotlib.pyplot as plt

def getModel(model_type, inchannels=30):
    if model_type == 'alexnet':
        model = AlexNet(input_channels=inchannels)
    elif model_type == 'lenet':
        model = LeNet(input_channels=inchannels)
    elif model_type == 'mobilenet':
        model = MobileNet(input_channels=inchannels)
    elif model_type == 'vgg':
        model = vgg16_bn(input_channels=inchannels)
    else:
        raise TypeError('输入的model_type不存在')
    return model

def warmUP(model, x, repeat=100):
    with torch.no_grad():
        for _ in range(repeat):
            _ = model(x)

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

def getGPULatency(model, x, repeat, starter, ender):
    model.eval()
    timings = np.zeros((repeat, 1))
    with torch.no_grad():
        for i in range(repeat):
            starter.record()
            _ = model(x)
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[i] = curr_time

    timings = deleteOutliersByIQR(timings)
    ti = np.mean(timings).item()
    return ti

def getCPULatency(model, x, repeat):
    model.eval()
    t_list = []
    with torch.no_grad():
        for i in range(repeat):
            sta = time.perf_counter()
            model(x)
            end = time.perf_counter()
            t_list.append(end - sta)

    t_list = deleteOutliersByIQR(t_list)
    ti = np.mean(t_list).item() * 1000
    return ti

def deleteOutliersByIQR(x_list, alpha=0.25):
    q1 = np.percentile(x_list, 25)
    q3 = np.percentile(x_list, 75)
    iqr = q3 - q1
    lower_bound = q1 - alpha * iqr
    upper_bound = q3 + alpha * iqr
    correct = np.where((x_list >= lower_bound) & (x_list <= upper_bound))
    return x_list[correct]

def monitorCPUInfo(N = 1000):
    def getFLOPS():
        # 定义矩阵的大小（越大计算越精确）
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)

        # 记录开始时间
        start_time = time.perf_counter()

        # 执行矩阵乘法
        np.dot(A, B)

        # 记录结束时间
        end_time = time.perf_counter()

        # 计算所用时间
        elapsed_time = end_time - start_time

        # 计算浮点运算次数：矩阵乘法的FLOPS计算公式为 2 * N^3
        num_operations = 2 * (N ** 3)
        flops = num_operations / elapsed_time

        return flops

    mem = psutil.virtual_memory()
    latency_threshold = (1000 - 10 * psutil.cpu_percent(interval=1)) / 1000.
    memory_threshold = mem.free
    flops = getFLOPS()
    return latency_threshold, memory_threshold, flops


