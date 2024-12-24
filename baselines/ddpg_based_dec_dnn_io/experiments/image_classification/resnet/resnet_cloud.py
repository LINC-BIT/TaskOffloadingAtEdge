import os
import argparse
import torch
import time
import multiprocessing as mp
from multiprocessing import Manager, cpu_count
from util.data.log import logger
import logging
from net.server import MultiMoniterDeviceInfoServer, MultiMoniterBWupServer, MultiMoniterBWdownServer, MainServer, DefaultDeployServer, TranmitModelServer
from net.util import closeDevices
from resnet_utils import resnet18_branchynet_cifar
from util.data.util import create_dir
from util.model.util import ModelProfiler
from data.image_classification.cifar100 import CIFAR100Dataloader
from ddpg.train import Trainer


parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, help='ip地址', default='10.1.114.109')
parser.add_argument('--port', type=int, help='端口号', default=9999)
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset [cifar10, cifar100]')
parser.add_argument('--cloud_device', type=str, help='使用设备', default='cpu')
args = parser.parse_args()

logging.getLogger('apscheduler').setLevel(logging.WARNING)

def start_monitor_bwup(devices, bws):
    monitor_ser = MultiMoniterBWupServer(devices, bws)
    monitor_ser.start()
    # for i, bw in enumerate(bws):
    #     logger.info(f"bandwidth monitor-{i} : get up bandwidth value : {bw:.3f} MB/s")

def start_monitor_bwdown(devices, bws):
    monitor_cli = MultiMoniterBWdownServer(devices, bws)
    monitor_cli.start()
    # for i, bw in enumerate(bws):
    #     logger.info(f"bandwidth monitor-{i} : get down bandwidth value : {bw:.3f} MB/s")

def start_monitor_device_info(devices, core_num, flops_array):
    monitor_cli = MultiMoniterDeviceInfoServer(devices, core_num, flops_array)
    monitor_cli.start()

def scheduler_for_bandwidth_monitor(fn, devices, bws, flag, sta_opt_flag):
    try:
        # 创建调度器
        while flag.value == 0:
            if sta_opt_flag.value == 0:
                fn(devices, bws)
                time.sleep(3)
    except Exception:
        pass

def scheduler_for_device_monitor(fn, devices, core_num, flops_array, flag, sta_opt_flag):
    # 创建调度器
    # try:
        while flag.value == 0:
            if sta_opt_flag.value == 0:
                fn(devices, core_num, flops_array)
                logger.info([v for v in flops_array])
                time.sleep(15)
    # except Exception:
    #     pass

def scheduler_optimize_deployment(trainer, emax, tmax, flag, sta_opt, running_flag):
    # try:
        while flag.value == 0:
            if running_flag.value == 1:
                sta_opt.value = 0
                trainer.start(emax, tmax)
                # logger.info(dec['info'])
                sta_opt.value = 1
                time.sleep(40)
    # except Exception:
    #     pass

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    cv_task = 'image_classification'
    dataset_name = args.dataset
    model_name = 'resnet'
    # method = 'legodnn'
    cloud_device = args.cloud_device
    model_input_size = (128, 3, 32, 32)

    root_path = os.path.join(args.path, cv_task,
                             model_name, dataset_name)

    create_dir(root_path)
    model_path = root_path + '/resnet.pth'
    num_classes= 100
    test_sample_num = 100
    acc_thres = 0.7
    core_num = cpu_count()
    batchsize = 4
    emax = 10
    tmax = 50
    ip = args.ip
    port = args.port
    bwup_run = mp.Value('b', 0)
    device_monitor_run = mp.Value('b', 0)
    running_flag = mp.Value('b', 1)
    sta_opt_flag = mp.Value('b', 0)

    # checkpoint = '/data/gxy/legodnn-auto-on-cv-models/cv_task_model/image_classification/cifar100/resnet18/2024-10-14/20-15-10/resnet18.pth'
    model, exit_points = resnet18_branchynet_cifar(num_classes, cloud_device)
    state_dict = torch.load(model_path, map_location=cloud_device)
    model.load_state_dict(state_dict)

    bwup_devices = Manager().dict()
    listener = MainServer(ip, 6666, bwup_devices)
    listener.start()
    listener.join()
    bwups = mp.Array('d', [0. for _ in range(len(bwup_devices['clients']))])

    de_info_devices = Manager().dict()
    listener = MainServer(ip, 6668, de_info_devices)
    listener.start()
    listener.join()
    flops_array = mp.Array('d', [0. for _ in range(1 + len(bwup_devices['clients']))])

    di = Manager().dict()
    listener = MainServer(ip, port, di)
    listener.start()
    listener.join()

    mp.Process(target=scheduler_for_bandwidth_monitor, args=(start_monitor_bwup, bwup_devices, bwups, bwup_run, sta_opt_flag)).start()
    mp.Process(target=scheduler_for_device_monitor, args=(start_monitor_device_info, de_info_devices, core_num, flops_array, device_monitor_run, sta_opt_flag)).start()

    while not all(v > 0 for v in flops_array):
        pass
        # print([v for v in flops_array])

    train_loader, test_loader = CIFAR100Dataloader(test_batch_size=model_input_size[0])

    p = ModelProfiler(model, model_input_size, exit_points, root_path, test_loader, cloud_device)
    p.profile_model()

    t = TranmitModelServer(di, model_path)
    t.start()

    dec = Manager().dict()
    trainer = Trainer(de_info_devices, model, exit_points, model_input_size, acc_thres, flops_array, bwups, dec,
                      test_loader, root_path, core_num, cloud_device, batchsize, root_path)

    if not os.path.exists(os.path.join(root_path, 'ddpg.pth')):
        trainer.start(100, 100)
        trainer.agent.save_model()
        sta_opt_flag.value = 1
        running_flag.value = 0
    else:
        sta_opt_flag.value = 0
        running_flag.value = 1

    deploy_flag = mp.Value('b', 0)
    mp.Process(target=scheduler_optimize_deployment,
               args=(trainer, emax, tmax, deploy_flag, sta_opt_flag, running_flag)).start()

    server = DefaultDeployServer(di, root_path, model, dec, test_loader, sta_opt_flag, running_flag, cloud_device)
    server.start()

    bwup_run.value = 1
    device_monitor_run.value = 1
    sta_opt_flag.value = 1
    running_flag.value = 0
    deploy_flag.value = 1

    closeDevices(di)
    closeDevices(de_info_devices)
    closeDevices(bwup_devices)
