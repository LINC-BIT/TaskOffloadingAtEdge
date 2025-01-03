import os
import argparse
import torch
import torch.nn as nn
import time
import multiprocessing as mp
from multiprocessing import Manager
import logging
from legodnn.utils.common.log import logger

from legodnn import BlockExtractor
from legodnn.block_detection.model_topology_extraction import topology_extraction
from legodnn.presets.common_detection_manager_1204_new import CommonDetectionManager
from legodnn.model_manager.common_model_manager import CommonModelManager
from legodnn.presets.auto_block_manager import AutoBlockManager
from legodnn.utils.dl.common.model import get_model_size
from cv_task.datasets.image_classification.cifar_dataloader import CIFAR100Dataloader
from cv_task.image_classification.cifar.models import resnet18
from offload.net.server import MainServer, ProfilesChecker, ProfilesRecver, MultiMoniterBWupServer, MultiMoniterBWdownServer, getClientNames, closeDevices, MultiMoniterDeviceInfoServer
from offload.deployment import MultiOptimalRuntime, BlockProfiler
from offload.experiments.image_classification.resnet18.resnet18_utils import getModelRootAndEnd, ResNetDeployServer, setOriginalBlock


parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, help='ip地址', default='10.1.114.109')
parser.add_argument('--port', type=int, help='端口号', default=9999)
parser.add_argument('--path', type=str, help='保存路径', default='./results/legodnn')
parser.add_argument('--edge_device', type=str, help='使用设备', default='cpu')
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

def scheduler_for_bandwidth_monitor(fn, devices, bws, flag):
    try:
        # 创建调度器
        while flag.value == 0:
            fn(devices, bws)
            time.sleep(3)
    except Exception:
        pass

def scheduler_for_device_monitor(fn, devices, latency_threshold, memory_threshold, flops_array, flag):
    # 创建调度器
    try:
        while flag.value == 0:
            fn(devices, latency_threshold, memory_threshold, flops_array)
            time.sleep(3)
    except Exception:
        pass

def scheduler_optimize_deployment(opt, flag, sta_opt):
    try:
        while flag.value == 0:
            if sta_opt.value == 0:
                opt.update_model()
                sta_opt.value = 1
            time.sleep(1)
    except Exception:
        pass

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    cv_task = 'image_classification'
    dataset_name = 'cifar100'
    model_name = 'resnet18'
    method = 'legodnn'
    edge_device = args.edge_device
    cloud_device = args.cloud_device
    compress_layer_max_ratio = 0.2
    model_input_size = (1, 3, 32, 32)
    block_sparsity = [0.0, 0.2, 0.4, 0.6, 0.8]

    root_path = os.path.join(args.path, cv_task,
                             model_name + '_' + dataset_name + '_' + str(compress_layer_max_ratio).replace('.', '-'))

    compressed_blocks_dir_path = root_path + '/compressed'
    trained_blocks_dir_path = root_path + '/trained'
    descendant_models_dir_path = root_path + '/descendant'
    offload_dir_path = root_path + '/offload'
    block_training_max_epoch = 65
    test_sample_num = 100
    ip = args.ip
    port = args.port
    bwup_run = mp.Value('b', 0)
    bwdown_run = mp.Value('b', 0)
    device_monitor_run = mp.Value('b', 0)
    deploy_flag = mp.Value('b', 0)

    checkpoint = '/data/zcr/legodnn/cv_task_model/image_classification/cifar100/resnet18/2024-12-02/12-50-06/resnet18.pth'
    teacher_model = resnet18(num_classes=100).to(cloud_device)
    teacher_model.load_state_dict(torch.load(checkpoint)['net'])

    print('\033[1;36m-------------------------------->    BUILD LEGODNN GRAPH\033[0m')
    model_graph = topology_extraction(teacher_model, model_input_size, device=cloud_device, mode='unpack')
    model_graph.print_ordered_node()

    print('\033[1;36m-------------------------------->    START BLOCK DETECTION\033[0m')
    detection_manager = CommonDetectionManager(model_graph, max_ratio=compress_layer_max_ratio)
    detection_manager.detection_all_blocks()
    detection_manager.print_all_blocks()

    model_manager = CommonModelManager()
    block_manager = AutoBlockManager(block_sparsity, detection_manager, model_manager)

    print('\033[1;36m-------------------------------->    START BLOCK EXTRACTION\033[0m')
    # block_extractor = BlockExtractor(teacher_model, block_manager, compressed_blocks_dir_path, model_input_size, 'cpu')
    # block_extractor.extract_all_blocks()

    root, end, rest_frame = getModelRootAndEnd(trained_blocks_dir_path, cloud_device)
    # frame = torch.load(os.path.join(compressed_blocks_dir_path, 'model_frame.pt')).to('cpu')
    # x = [torch.rand((1, 3, 32, 32)) for i in range(100)]
    _, test_loader = CIFAR100Dataloader(test_batch_size=128, num_workers=1)

    # test_loader = iter(test_loader)

    blocks_name = block_manager.get_blocks_id()
    block_dict = {}
    old_block_dict = {}
    for id, sp in zip([0,1,2,3,4,5,6,7], [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]):
        block = block_manager.get_block_from_file(
            os.path.join(trained_blocks_dir_path, block_manager.get_block_file_name(blocks_name[id], sp)),
            'cpu').to('cpu')
        nblock = setOriginalBlock(rest_frame, block)
        block_dict[blocks_name[id]] = nblock
        old_block_dict[blocks_name[id]] = block

    # x, y = next(test_loader)
    #
    # frame(x)
    # x = frame.conv1(x)
    model = nn.Sequential()
    model.add_module('root', root)
    for k, v in block_dict.items():
        # x = v(x)
        model.add_module(k, v)
    model.add_module('end', end)

    # old_model = nn.Sequential()
    # old_model.add_module('root', root)
    # for k, v in old_block_dict.items():
    #     # x = v(x)
    #     old_model.add_module(k, v)
    # old_model.add_module('end', end)

    # for id in range(len(model)):
    #     t1x = model[id](x)
    #     t2x = old_model[id](x)
    #     x = t2x
    # print('a')
    # x = teacher_model(x)
    #
    # pred = torch.argmax(x, dim=1)
    # correct = torch.sum(pred == y).item()
    # print('a')

    a = model_manager.get_model_acc(model, test_loader, 'cpu')
    print(a)