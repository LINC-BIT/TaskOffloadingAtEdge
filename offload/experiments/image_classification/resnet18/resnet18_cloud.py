import os
import argparse
import torch
import time
import multiprocessing as mp
from multiprocessing import Manager
import logging
from legodnn.utils.common.log import logger
from legodnn.utils.common.file import create_dir
from legodnn.block_detection.model_topology_extraction import topology_extraction
from legodnn.presets.common_detection_manager_1204_new import CommonDetectionManager
from legodnn.model_manager.common_model_manager import CommonModelManager
from legodnn.presets.auto_block_manager import AutoBlockManager
from legodnn.utils.dl.common.model import get_model_size
from cv_task.datasets.image_classification.cifar_dataloader import CIFAR100Dataloader
from cv_task.image_classification.cifar.models import resnet18
from offload.net.server import MainServer, ProfilesChecker, MultiMoniterBWupServer, MultiMoniterBWdownServer, getClientNames, closeDevices, MultiMoniterDeviceInfoServer
from offload.deployment import MultiOptimalRuntime
from offload.experiments.image_classification.resnet18.resnet18_utils import getModelRootAndEnd, ResNetDeployServer


parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, help='ip地址', default='10.1.114.109')
parser.add_argument('--port', type=int, help='端口号', default=9999)
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset [cifar10, cifar100]')
parser.add_argument('--acc_thres', default=0.763, type=float, help='准确率阈值')
parser.add_argument('--compress_layer_max_ratio', default=0.2, type=float, help='LegoDNN压缩率')
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

def start_monitor_device_info(devices, latency_th, memory_th, flops_array):
    monitor_cli = MultiMoniterDeviceInfoServer(devices, latency_th, memory_th, flops_array)
    monitor_cli.start()

# def scheduler_for_bandwidth_monitor(fn, devices, bws, flag, sta_opt):
#     try:
#         # 创建调度器
#         while flag.value == 0:
#             if sta_opt.value == 0:
#                 fn(devices, bws)
#                 time.sleep(3)
#     except Exception:
#         pass

def scheduler_for_device_monitor(fn, devices, latency_threshold, memory_threshold, flops_array, flag, sta_opt):
    # 创建调度器
    try:
        while flag.value == 0:
            if sta_opt.value == 0:
                fn(devices, latency_threshold, memory_threshold, flops_array)
                # logger.info([f for f in flops_array])
                time.sleep(3)
    except Exception:
        pass

def scheduler_optimize_deployment(opt, flag, sta_opt, running_flag, deploy_infos):
    try:
        while flag.value == 0:
            if running_flag.value == 1:
                sta_opt.value = 0
                time.sleep(10)
                opt.update_model()
                sta_opt.value = 1
                logger.info(deploy_infos['infos'])
                time.sleep(30)
    except Exception:
        pass

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    cv_task = 'image_classification'
    dataset_name = args.dataset
    model_name = 'resnet18'
    method = 'legodnn'
    cloud_device = args.cloud_device
    compress_layer_max_ratio = args.compress_layer_max_ratio
    model_input_size = (128, 3, 32, 32)
    block_sparsity = [0.0, 0.2, 0.4, 0.6, 0.8]
    acc_thres = args.acc_thres

    root_path = os.path.join('./results', method, cv_task,
                             model_name + '_' + dataset_name + '_' + str(compress_layer_max_ratio).replace('.', '-'))

    compressed_blocks_dir_path = root_path + '/compressed'
    trained_blocks_dir_path = root_path + '/trained'
    descendant_models_dir_path = root_path + '/descendant'
    offload_dir_path = root_path + '/offload'
    ip = args.ip
    port = args.port
    device_monitor_run = mp.Value('b', 0)
    deploy_flag = mp.Value('b', 0)
    start_opt = mp.Value('b', 0)
    running_flag = mp.Value('b', 1)

    create_dir(offload_dir_path)
    
    if dataset_name == 'cifar100':
        teacher_model = resnet18(num_classes=100).to(cloud_device)
        _, test_loader = CIFAR100Dataloader(test_batch_size=model_input_size[0])
        # teacher_model.load_state_dict(torch.load(checkpoint)['net'])
    else:
        teacher_model = resnet18(num_classes=10).to(cloud_device)
        _, test_loader = CIFAR100Dataloader(test_batch_size=model_input_size[0])

    di = Manager().dict()
    listener = MainServer(ip, port, di)
    listener.start()
    listener.join()

    bwup_devices = Manager().dict()
    listener = MainServer(ip, 6666, bwup_devices)
    listener.start()
    listener.join()
    bwups = mp.Array('d', [0. for _ in range(len(di['clients']))])

    bwdown_devices = Manager().dict()
    listener = MainServer(ip, 6667, bwdown_devices)
    listener.start()
    listener.join()
    bwdowns = mp.Array('d', [0. for _ in range(len(di['clients']))])

    de_info_devices = Manager().dict()
    listener = MainServer(ip, 6668, de_info_devices)
    listener.start()
    listener.join()

    latency_th = mp.Array('d', [0. for _ in range(1 + len(di['clients']))])
    memory_th = mp.Array('d', [0. for _ in range(1 + len(di['clients']))])
    flops_array = mp.Array('d', [0. for _ in range(1 + len(di['clients']))])

    mp.Process(target=scheduler_for_device_monitor, args=(start_monitor_device_info, de_info_devices, latency_th, memory_th, flops_array, device_monitor_run, start_opt)).start()

    start_monitor_bwup(bwup_devices, bwups)
    start_monitor_bwdown(bwdown_devices, bwdowns)

    closeDevices(bwup_devices)
    closeDevices(bwdown_devices)

    print('\033[1;36m-------------------------------->    BUILD LEGODNN GRAPH\033[0m')
    model_graph = topology_extraction(teacher_model, model_input_size, device=cloud_device, mode='unpack')
    model_graph.print_ordered_node()

    print('\033[1;36m-------------------------------->    START BLOCK DETECTION\033[0m')
    detection_manager = CommonDetectionManager(model_graph, max_ratio=compress_layer_max_ratio)
    detection_manager.detection_all_blocks()
    detection_manager.print_all_blocks()

    model_manager = CommonModelManager()
    block_manager = AutoBlockManager(block_sparsity, detection_manager, model_manager)

    print('\033[1;36m-------------------------------->    TRANSMIT BLOCK PROFILES\033[0m')
    process = ProfilesChecker(trained_blocks_dir_path, di, block_manager)
    process.start()
    process.join()

    print('\033[1;36m-------------------------------->    START BLOCK SELECTING\033[0m')
    client_names = getClientNames(di)
    deploy_infos = Manager().dict()
    optimal_runtime = MultiOptimalRuntime(trained_blocks_dir_path, model_input_size, block_manager,
                                          model_manager, deploy_infos, bwdowns=bwdowns, bwups=bwups, l_ths=latency_th,
                                          m_ths=memory_th, f_array=flops_array, acc_thres=acc_thres, device=cloud_device, clients_name=client_names)
    model_size_min = get_model_size(torch.load(os.path.join(trained_blocks_dir_path, 'model_frame.pt')))
    model_size_max = get_model_size(teacher_model)
    optimal_runtime.setCurInferThresholds(4e-1, model_size_max / 4)

    mp.Process(target=scheduler_optimize_deployment, args=[optimal_runtime, deploy_flag, start_opt, running_flag, deploy_infos]).start()
    # print(deploy_infos[0])

    print('\033[1;36m-------------------------------->    START BLOCK INFERENCE\033[0m')
    frame = getModelRootAndEnd(trained_blocks_dir_path, cloud_device)
    x = []

    d_server = ResNetDeployServer(test_loader, frame, deploy_infos, block_manager, trained_blocks_dir_path,
                                  offload_dir_path, cloud_device, start_opt, running_flag, di, bwdowns, bwups)
    d_server.start()

    device_monitor_run.value = 1
    deploy_flag.value = 1

    # print(res)
    time.sleep(3)

    closeDevices(di)
    closeDevices(bwup_devices)
    closeDevices(bwdown_devices)
    closeDevices(de_info_devices)