import os
import argparse
import time
import sys
from multiprocessing import Manager
import multiprocessing as mp
import torch
import logging

sys.path.append(os.getcwd())
from experiments.image_classification.resnet.resnet_utils import resnet18_branchynet_cifar
from net.util import close_conn
from net.client import MultiMoniterDeviceInfoClient, MultiMoniterBWupClient, MultiMoniterBWdownClient, MainClient, TranmitModelClient, DefaultClient

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, help='ip地址', default='10.1.114.109')
parser.add_argument('--port', type=int, help='端口号', default=9999)
parser.add_argument('--path', type=str, help='保存路径', default='./results')
parser.add_argument('--edge_device', type=str, help='使用设备', default='cpu')
parser.add_argument('--cloud_device', type=str, help='使用设备', default='cpu')
args = parser.parse_args()

logging.getLogger('apscheduler').setLevel(logging.WARNING)

def start_monitor_bwup(client):
    monitor_ser = MultiMoniterBWupClient(client)
    monitor_ser.start()

def start_monitor_bwdown(client):
    monitor_cli = MultiMoniterBWdownClient(client)
    monitor_cli.start()

def start_monitor_device_info(client, core_num):
    monitor_cli = MultiMoniterDeviceInfoClient(client, core_num)
    monitor_cli.start()

def scheduler_for_bandwidth_monitor(fn, client, flag):
    # 创建调度器
    try:
        while flag.value == 0:
            fn(client)
            time.sleep(3)
    except Exception:
        pass

def scheduler_for_device_monitor(fn, client, flag, core_num):
    # 创建调度器
    try:
        while flag.value == 0:
            fn(client, core_num)
            time.sleep(3)
    except Exception:
        pass

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    cv_task = 'image_classification'
    dataset_name = 'cifar100'
    model_name = 'resnet'
    # method = 'legodnn'
    core_num = 20
    edge_device = args.edge_device
    cloud_device = args.cloud_device

    root_path = os.path.join(args.path, cv_task,
                             model_name, dataset_name)

    num_classes = 100
    acc_thres = 0.7
    batchsize = 4
    emax = 100
    tmax = 100
    ip = args.ip
    port = args.port
    bwup_run = mp.Value('b', 0)
    device_monitor_run = mp.Value('b', 0)

    bwup_client = Manager().dict()
    listener = MainClient(ip, 6666, bwup_client)
    listener.start()
    listener.join()

    device_info_client = Manager().dict()
    listener = MainClient(ip, 6668, device_info_client)
    listener.start()
    listener.join()

    client = Manager().dict()
    listener = MainClient(ip, port, client)
    listener.start()
    listener.join()

    mp.Process(target=scheduler_for_bandwidth_monitor, args=(start_monitor_bwup, bwup_client, bwup_run)).start()
    mp.Process(target=scheduler_for_device_monitor, args=(start_monitor_device_info, device_info_client, device_monitor_run, core_num)).start()

    model_path = root_path + '/resnet.pth'
    t = TranmitModelClient(client, model_path)
    t.start()

    model, _ = resnet18_branchynet_cifar(num_classes, edge_device)
    state_dict = torch.load(model_path, map_location=cloud_device)
    model.load_state_dict(state_dict)

    c = DefaultClient(client, model, core_num, edge_device)
    c.start()

    bwup_run.value = 1
    device_monitor_run.value = 1

    close_conn(client['client'])