import math

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os
from offload.predictor.data_obtainer import getDatasets
import pickle
from offload.predictor.utils import *
from offload.model.MobileNet import InvertedResidual
from cv_task.image_classification.cifar.models.resnet import Bottleneck, BasicBlock
from typing import Iterable

def getRegression(type='poly'):
    if type == 'tree':
        return DecisionTreeRegressor(max_depth=10, random_state=42)
    elif type == 'poly':
        return LinearRegression()
    else:
        raise TypeError(f'不支持{type}类型')

def getLatency(rootpath, model, x, device, is_gpu):
    latency = 0.
    for layer in model.children():
        if not isinstance(layer, nn.ReLU) and not isinstance(layer, nn.Dropout) and not isinstance(layer, nn.Flatten) and not isinstance(layer, nn.ReLU6):
            if isinstance(layer, nn.Sequential) or isinstance(layer, Bottleneck) or isinstance(layer, BasicBlock) \
                    or isinstance(layer, InvertedResidual) or isinstance(layer,Iterable):
                latency += getLatency(rootpath, layer, x, device, is_gpu)
            else:
                pred = getPredictor(rootpath, layer, device, is_gpu)
                feats = getFeatures(x, layer)
                if isinstance(pred, list):
                    pred, poly_f = pred
                    feats = poly_f.transform(feats)

                latency += max(pred.predict(feats)[0], 0)
                x = layer(x)
    return latency

def trainModel(rootpath, layer, device, is_gpu):
    path = os.path.join(rootpath, f'models/{device}')
    addr = "gpu" if is_gpu else "cpu"
    if isinstance(layer, nn.Conv2d):
        if layer.groups == 1:
            path = os.path.join(path, f'conv_{addr}.pkl')
            print(f'开始进行卷积层在设备{device}上的延迟的线性回归')
        else:
            path = os.path.join(path, f'dwconv_{addr}.pkl')
            print(f'开始进行DW卷积层在设备{device}上的延迟的线性回归')
    elif isinstance(layer, nn.Linear):
        path = os.path.join(path, f'linear_{addr}.pkl')
        print(f'开始进行线性层在设备{device}上的延迟的线性回归')
    elif isinstance(layer, nn.MaxPool2d):
        path = os.path.join(path, f'maxpool_{addr}.pkl')
        print(f'开始进行最大池化层在设备{device}上的延迟的线性回归')
    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        path = os.path.join(path, f'avgpool_{addr}.pkl')
        print(f'开始进行平均池化层在设备{device}上的延迟的线性回归')
    elif isinstance(layer, nn.BatchNorm2d):
        path = os.path.join(path, f'batchnorm_{addr}.pkl')
        print(f'开始进行BN层在设备{device}上的延迟的线性回归')
    else:
        raise TypeError('目前不支持所输入层')

    df = getDatasets(rootpath, layer, device, is_gpu)
    if 'FLOPs' in df.columns:
        df['FLOPs'] = df['FLOPs'].apply(lambda x: math.log2(x))
    y = df['latency'].to_numpy()
    x = df.iloc[:,:-1].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=111)

    model_type = 'poly'
    if model_type == 'poly':
        if isinstance(layer, nn.Conv2d) and layer.groups == 1:
            degree = 1
        else:
            degree = 1
        poly_f = PolynomialFeatures(degree=degree)
        x_train = poly_f.fit_transform(x_train)
        x_test = poly_f.transform(x_test)

    reg = getRegression(type=model_type)
    reg.fit(x_train, y_train)

    y_pred = reg.predict(x_test)
    print('训练结果：')
    print(f'MSE={mean_squared_error(y_test, y_pred)}')
    if model_type == 'linear':
        print(f'R2 Score={r2_score(y_test, y_pred)}')

    with open(path, 'wb') as f:
        if model_type == 'poly':
            pickle.dump([reg, poly_f], f)
        else:
            pickle.dump(reg, f)

def getPredictor(rpath, layer, device, is_gpu):
    # rpath = os.path.join(rootpath, 'predictor')
    addr = "gpu" if is_gpu else "cpu"
    if isinstance(layer, nn.Conv2d):
        if layer.groups == 1:
            path = os.path.join(rpath, f'models/{device}/conv_{addr}.pkl')
        else:
            path = os.path.join(rpath, f'models/{device}/dwconv_{addr}.pkl')
    elif isinstance(layer, nn.Linear):
        path = os.path.join(rpath, f'models/{device}/linear_{addr}.pkl')
    elif isinstance(layer, nn.MaxPool2d):
        path = os.path.join(rpath, f'models/{device}/maxpool_{addr}.pkl')
    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        path = os.path.join(rpath, f'models/{device}/avgpool_{addr}.pkl')
    elif isinstance(layer, nn.BatchNorm2d):
        path = os.path.join(rpath, f'models/{device}/batchnorm_{addr}.pkl')
    else:
        raise TypeError('目前不支持所输入层')
    if not os.path.exists(path):
        trainModel(rpath, layer, device, is_gpu)
    with open(path, 'rb') as f:
        reg = pickle.load(f)
    return reg

def getFeatures(data ,layer):
    if isinstance(layer, nn.Conv2d):
        if layer.groups == 1:
            flops = getFlopsOfConv2d(layer, data)
            flops = math.log2(flops)
            feat = (data.shape[2], layer.kernel_size[0], layer.stride[0], layer.in_channels, layer.out_channels, flops)
        else:
            flops = math.log2(getFlopsOfDwConv2d(layer, data))
            feat = (data.shape[2], layer.kernel_size[0], layer.stride[0], layer.in_channels, flops)
    elif isinstance(layer, nn.Linear):
        flops = math.log2(getFlopsOfLinear(layer, data))
        feat = (layer.in_features, layer.out_features, flops)
    elif isinstance(layer, nn.MaxPool2d):
        feat = (data.shape[2], layer.kernel_size, layer.stride, data.shape[1])
    elif isinstance(layer, nn.AdaptiveAvgPool2d):
        feat = (data.shape[2], layer.output_size[0], data.shape[1])
    elif isinstance(layer, nn.BatchNorm2d):
        feat = (data.shape[2], data.shape[1])
    else:
        raise TypeError('目前不支持所输入层')
    feat = np.array(feat).reshape((1, -1))
    return feat

if __name__ == '__main__':
    device = 'cloud'
    gpu = True
    path = '/data/zcr/legodnn/results/legodnn/image_classification/resnet18_cifar100_0-125/offload/'
    getPredictor(path , nn.Conv2d(2,2,2), device, gpu)
    getPredictor(path, nn.Conv2d(2,2,2,groups=2), device, gpu)
    getPredictor(path, nn.Linear(2,2), device, gpu)
    getPredictor(path, nn.AdaptiveAvgPool2d((2,2)), device, gpu)
    getPredictor(path, nn.MaxPool2d(2,2), device, gpu)
    getPredictor(path, nn.BatchNorm2d(2), device, gpu)






