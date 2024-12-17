import torch.nn as nn
import torch
from functools import singledispatch

@singledispatch
def getFlopsOfConv2d(*args) -> int:
    raise AttributeError('传入错误的参数')

@getFlopsOfConv2d.register(nn.Conv2d)
def _(conv:nn.Conv2d, x:torch.Tensor):
    in_channel = conv.in_channels
    out_channel = conv.out_channels
    kernel_size = conv.kernel_size[0]
    padding = conv.padding[0]
    stride = conv.stride[0]
    input_map = x.shape[2]
    output_map = (input_map - kernel_size + 2 * padding) // stride + 1
    flops = 2 * output_map * output_map * kernel_size * kernel_size * in_channel * out_channel
    return flops

@getFlopsOfConv2d.register(int)
def _(in_channel:int, out_channel:int, kernel_size:int, padding:int, stride:int, hw:int):
    input_map = hw
    output_map = (input_map - kernel_size + 2 * padding) // stride + 1
    flops = 2 * output_map * output_map * kernel_size * kernel_size * in_channel * out_channel
    return flops

@singledispatch
def getFlopsOfDwConv2d(*args) -> int:
    raise AttributeError('传入错误的参数')

@getFlopsOfDwConv2d.register(int)
def _(in_channel:int, kernel_size:int, padding:int, stride:int, hw:int): #Depthwise
    input_map = hw
    output_map = (input_map - kernel_size + 2 * padding) // stride + 1
    flops = 2 * output_map * output_map * kernel_size * kernel_size * in_channel
    return flops

@getFlopsOfDwConv2d.register(nn.Conv2d)
def _(conv_dw:nn.Conv2d, x:torch.Tensor): #Depthwise
    input_map = x.shape[2]
    kernel_size = conv_dw.kernel_size[0]
    padding = conv_dw.padding[0]
    stride = conv_dw.stride[0]
    in_channel = conv_dw.in_channels
    output_map = (input_map - kernel_size + 2 * padding) // stride + 1
    flops = 2 * output_map * output_map * kernel_size * kernel_size * in_channel
    return flops

@singledispatch
def getFlopsOfLinear(*args) -> int:
    return 0

@getFlopsOfLinear.register(int)
def _(in_feature:int, out_feature:int):
    return (2 * in_feature - 1) * out_feature

@getFlopsOfLinear.register(nn.Linear)
def _(linear:nn.Linear, x:torch.Tensor):
    in_feature = x.shape[1]
    out_feature = linear.out_features
    flops = ((2 * in_feature - 1) if linear.bias is None else (2 * in_feature)) * out_feature
    return flops

def binary_search(arr, target):
    """
    二分查找算法
    :param arr: 有序数组
    :param target: 目标元素
    :return: 目标元素的索引，如果不存在则返回-1
    """
    low = 0
    high = len(arr)
    while low < high:
        mid = (low + high) // 2
        if arr[mid] < target:
            low = mid + 1
        else:
            high = mid
    if target < arr[low]:
        low -= 1
    return low

def moduleDictForward(layer, x):
    for _, l in layer.items():
        if isinstance(l, nn.ModuleDict):
            x = moduleDictForward(l, x)
        else:
            x = l(x)
    return x