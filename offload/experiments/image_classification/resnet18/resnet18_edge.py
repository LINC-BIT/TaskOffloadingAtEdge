import os
import argparse
import time
import sys
from multiprocessing import Manager
import multiprocessing as mp
import torch
import logging

sys.path.append(os.getcwd())

from cv_task.image_classification.cifar.models import resnet18
from legodnn.block_detection.model_topology_extraction import topology_extraction
from legodnn.presets.common_detection_manager_1204_new import CommonDetectionManager
from legodnn.model_manager.common_model_manager import CommonModelManager
from legodnn.presets.auto_block_manager import AutoBlockManager
from offload.net.client import MainClient, ProfilesRecver, MultiMoniterBWupClient, MultiMoniterBWdownClient, MultiMoniterDeviceInfoClient
from offload.net.utils import close_conn
from offload.experiments.image_classification.resnet18.resnet18_utils import ResNetDeployClient, getModelRootAndEnd

parser = argparse.ArgumentParser()
parser.add_argument('--ip', type=str, help='ip地址', default='10.1.114.109')
parser.add_argument('--port', type=int, help='端口号', default=9999)
parser.add_argument('--dataset', default='cifar100', type=str, help='dataset [cifar10, cifar100]')
parser.add_argument('--compress_layer_max_ratio', default=0.2, type=float, help='LegoDNN压缩率')
parser.add_argument('--edge_device', type=str, help='使用设备', default='cpu')
args = parser.parse_args()

logging.getLogger('apscheduler').setLevel(logging.WARNING)

def start_monitor_bwup(client):
    monitor_ser = MultiMoniterBWupClient(client)
    monitor_ser.start()

def start_monitor_bwdown(client):
    monitor_cli = MultiMoniterBWdownClient(client)
    monitor_cli.start()

def start_monitor_device_info(client):
    monitor_cli = MultiMoniterDeviceInfoClient(client)
    monitor_cli.start()

def scheduler_for_bandwidth_monitor(fn, client, flag):
    # 创建调度器
    # try:
        while flag.value == 0:
            fn(client)
            time.sleep(3)
    # except Exception:
    #     pass

def scheduler_for_device_monitor(fn, client, flag):
    # 创建调度器
    # try:
        while flag.value == 0:
            fn(client)
            time.sleep(3)
    # except Exception:
    #     pass

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    cv_task = 'image_classification'
    dataset_name = args.dataset
    model_name = 'resnet18'
    method = 'legodnn'
    edge_device = args.edge_device
    compress_layer_max_ratio = args.compress_layer_max_ratio
    model_input_size = (1, 3, 32, 32)
    block_sparsity = [0.0, 0.2, 0.4, 0.6, 0.8]
    edge_latency_threshold = mp.Value('d', 0.)
    edge_memory_threshold = mp.Value('d', 0.)

    root_path = os.path.join('./results', method, cv_task,
                             model_name + '_' + dataset_name + '_' + str(compress_layer_max_ratio).replace('.', '-'))

    compressed_blocks_dir_path = root_path + '/compressed'
    trained_blocks_dir_path = root_path + '/trained'
    descendant_models_dir_path = root_path + '/descendant'
    offload_dir_path = root_path + '/offload'
    test_sample_num = 100
    ip = args.ip
    port = args.port
    device_monitor_run = mp.Value('b', 0)

    teacher_model = resnet18(num_classes=100).to(edge_device)

    client = Manager().dict()
    listener = MainClient(ip, port, client)
    listener.start()
    listener.join()

    bwup_client = Manager().dict()
    listener = MainClient(ip, 6666, bwup_client)
    listener.start()
    listener.join()

    bwdown_client = Manager().dict()
    listener = MainClient(ip, 6667, bwdown_client)
    listener.start()
    listener.join()

    device_info_client = Manager().dict()
    listener = MainClient(ip, 6668, device_info_client)
    listener.start()
    listener.join()

    # mp.Process(target=scheduler_for_bandwidth_monitor, args=(start_monitor_bwup, bwup_client, bwup_run)).start()
    # mp.Process(target=scheduler_for_bandwidth_monitor, args=(start_monitor_bwdown, bwdown_client, bwdown_run)).start()
    mp.Process(target=scheduler_for_device_monitor, args=(start_monitor_device_info, device_info_client, device_monitor_run)).start()

    start_monitor_bwup(bwup_client)
    start_monitor_bwdown(bwdown_client)

    close_conn(bwup_client['client'])
    close_conn(bwdown_client['client'])
    print('\033[1;36m-------------------------------->    BUILD LEGODNN GRAPH\033[0m')
    model_graph = topology_extraction(teacher_model, model_input_size, device=edge_device, mode='unpack')
    model_graph.print_ordered_node()

    print('\033[1;36m-------------------------------->    START BLOCK DETECTION\033[0m')
    detection_manager = CommonDetectionManager(model_graph, max_ratio=compress_layer_max_ratio)
    detection_manager.detection_all_blocks()
    detection_manager.print_all_blocks()

    model_manager = CommonModelManager()
    block_manager = AutoBlockManager(block_sparsity, detection_manager, model_manager)

    print('\033[1;36m-------------------------------->    CHECK BLOCK PROFILES\033[0m')
    process = ProfilesRecver(trained_blocks_dir_path, client)
    process.start()
    process.join()

    print('\033[1;36m-------------------------------->    START BLOCK INFERENCE\033[0m')
    _, _, rest_frame = getModelRootAndEnd(trained_blocks_dir_path, edge_device)
    d_client = ResNetDeployClient(block_manager, rest_frame, trained_blocks_dir_path, edge_device, client)
    d_client.start()

    device_monitor_run.value = 1

    time.sleep(3)

    close_conn(client['client'])
    close_conn(device_info_client['client'])