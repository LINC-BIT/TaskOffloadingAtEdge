import abc
import os.path
from multiprocessing import Process
from offload.net.utils import *
from offload.inference_utils import monitorCPUInfo
import torch
import time
from collections import deque
from offload.deployment import getPartitionPoint, getProfiles, divideModel
from offload.inference_utils import getModel, Inference
from legodnn.utils.common.file import create_dir
from legodnn.utils.dl.common.model import save_model, ModelSaveMethod
from legodnn.utils.common.log import logger
import yaml
import pandas as pd
import torch.nn as nn
from copy import deepcopy

class Client(Process):
    def __init__(self, ip, port):
        super().__init__()
        self.server_ip = ip
        self.server_port = port
        self.conn = None

    def _createClient(self):
        self.conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)


    @abc.abstractmethod
    def startClient(self, *args):
        pass

    def run(self) -> None:
        self.startClient()

# For 1 cloud 1 edge
class LegodnnProfilesClient(Client):
    def __init__(self, ip, port, trained_path=''):
        super().__init__(ip, port)
        self.trained_path = trained_path

    def startClient(self):
        while True:
            try:
                self._createClient()
                self.conn.connect((self.server_ip, self.server_port))
                if not os.path.exists(self.trained_path):
                    logger.info('开始下载训练数据')
                    create_dir(self.trained_path)
                    send_message(self.conn, 'Block Request', msg='请求下载已训练的块', show=True)
                    block_nums = get_short_data(self.conn)

                    send_message(self.conn, 'break', False)

                    for _ in range(block_nums):
                        block, block_name = get_data(self.conn)
                        save_model(block, os.path.join(self.trained_path, block_name), ModelSaveMethod.FULL)

                    csv, csv_name = get_data(self.conn)
                    csv.to_csv(os.path.join(self.trained_path, csv_name), index=False)

                    y, yaml_name = get_data(self.conn)
                    with open(os.path.join(self.trained_path, yaml_name), 'w') as f:
                        yaml.dump(y, f)
                    logger.info('成功获得数据')
                else:
                    logger.info('无需接收数据')
                    send_message(self.conn, 'Fine', msg='无需下载', show=False)

                close_conn(self.conn)
                break
            except ConnectionRefusedError:
                logger.info('重试连接中...')
            time.sleep(1)

class LegodnnSendingClient(Client):
    def __init__(self, ip, port, offload_path=''):
        super().__init__(ip, port)
        self.offload_path = offload_path

    def startClient(self):
        while True:
            try:
                self._createClient()
                self.conn.connect((self.server_ip, self.server_port))
                s = get_message(self.conn, show=True)

                if s == 'Require edge profiles':
                    logger.info('开始上传信息')

                    df = pd.read_csv(os.path.join(self.offload_path, 'edge-blocks-metrics.csv'))
                    pack = [df, 'edge-blocks-metrics.csv']
                    send_data(self.conn, pack, 'Edge Metrics', True)

                    f = open(os.path.join(self.offload_path, 'edge-teacher-model-metrics.yaml'))
                    metrics = yaml.load(f, yaml.Loader)
                    pack = [metrics, 'edge-teacher-model-metrics.yaml']
                    send_data(self.conn, pack, 'edge-teacher-model-metrics.yaml', True)

                    logger.info('成功上传数据')
                else:
                    logger.info('无需上传数据')
                close_conn(self.conn)
                break
            except ConnectionRefusedError:
                logger.info('重试连接中...')
            time.sleep(1)

class DefaultClient(Client):
    def __init__(self, ip, port, bw, model_type='', root_path='', is_edge_gpu=True, is_cloud_gpu=True):
        super().__init__(ip, port)
        self.bw = bw
        self.model_type = model_type
        self.rootpath = root_path
        self.is_edge_gpu = is_edge_gpu
        self.is_cloud_gpu = is_cloud_gpu

    def setInput(self, x):
        self.x = x.to('cuda' if self.is_edge_gpu else 'cpu')

    def setOutput(self, o):
        self.output = o

    def setLatencyVar(self, l):
        self.latency = l

    def startClient(self, repeat=10):
        assert self.x is not None, '未设定输入'
        model = getModel(self.model_type, inchannels=self.x.shape[1])
        profiles = getProfiles(self.rootpath, model, self.x.shape, self.is_edge_gpu, self.is_cloud_gpu, self.bw.value)

        for _ in range(repeat):
            self._createClient()
            self.conn.connect((self.server_ip, self.server_port))
            send_short_data(self.conn, self.model_type, 'Model type')
            print(f'Repeat-{_+1}:')
            print(f'网速:{self.bw.value}MB/s')

            point = getPartitionPoint(profiles, self.bw.value)

            send_short_data(self.conn, point, 'Partition Point')

            edge_model, _ = divideModel(model, point)
            edge_model = edge_model.to('cuda' if self.is_edge_gpu else 'cpu')

            self.conn.recv(40)

            x, edge_latency = Inference(edge_model, self.x, self.is_edge_gpu)

            print(f'边缘推理延迟:{edge_latency}ms')

            if len(model) != point:
                send_data(self.conn, x, 'IR')

                self.conn.sendall('break'.encode())

                trans_latency = get_short_data(self.conn)
                print(f'传输延迟:{trans_latency}ms')

                self.conn.sendall('break'.encode())

                cloud_latency = get_short_data(self.conn)
                print(f'云推理延迟:{cloud_latency}ms')

                self.conn.sendall('break'.encode())

                output, _ = get_data(self.conn)
                output = output.view(-1).tolist()
                # for id, i in enumerate(output):
                #     self.output[id] = i
                self.latency.value += edge_latency + trans_latency + cloud_latency
            else:
                self.latency.value += edge_latency

            close_conn(self.conn)
            time.sleep(2)

        self.latency.value /= repeat
        print(f'成功获得结果')


class MoniterClient(Client):
    def __init__(self, ip, port, bw):
        super().__init__(ip, port)
        self.bw = bw

    def startClient(self):
        x = torch.rand((1, 3, 224, 224))
        gamma = 0.5
        while True:
            try:
                self._createClient()
                self.conn.connect((self.server_ip, self.server_port))
                send_data(self.conn, x)

                send_short_data(self.conn, 'break', show=False)

                if self.bw.value != 0:
                    bw = gamma * get_short_data(self.conn) + (1 - gamma) * self.bw.value
                    # bw = min(self.bw.value, get_short_data(self.conn))
                else:
                    bw = get_short_data(self.conn)

                if bw is not None:
                    self.bw.value = bw
                    close_conn(self.conn)
                    break
            except ConnectionRefusedError:
                # print('重试连接中...')
                pass
            time.sleep(1)

class MoniterBWupClient(Client):
    def __init__(self, ip, port, first=False):
        super().__init__(ip, port)
        self.first = first

    def startClient(self):
        x = torch.rand((1, 3, 224, 224))
        while True:
            try:
                self._createClient()
                self.conn.connect((self.server_ip, self.server_port))
                send_data(self.conn, x)

                close_conn(self.conn)
                break
            except ConnectionRefusedError:
                # print('重试连接中...')
                pass
            if not self.first:
                break
            time.sleep(1)

class MoniterBWdownClient(Client):
    def __init__(self, ip, port, first=False):
        super().__init__(ip, port)
        self.first = first

    def startClient(self):
        while True:
            try:
                self._createClient()
                self.conn.connect((self.server_ip, self.server_port))
                bw = get_bandwidth(self.conn)

                get_message(self.conn, show=False)  # 防止粘包

                send_short_data(self.conn, bw, 'Bandwidth', show=False)

                close_conn(self.conn)

                break
            except ConnectionRefusedError:
                # print('重试连接中...')
                pass
            if not self.first:
                break
            time.sleep(1)

class LegodnnDeployClient(Client):
    def __init__(self, ip, port, x, block_manager, trained_path, device, res):
        super().__init__(ip, port)
        self._block_manager = block_manager
        self._trained_path = trained_path
        self._blocks_name = None
        self._block_dict = None
        self._device = device
        self._x = x.to(self._device)
        self._res = res

    def _getblocks(self, block_id, sps):
        self._blocks_name = self._block_manager.get_blocks_id()
        self._block_dict = {}
        for id, sp in zip(block_id, sps):
            block = self._block_manager.get_block_from_file(
                os.path.join(self._trained_path,self._block_manager.get_block_file_name(self._blocks_name[id], sp)),
                self._device).to(self._device)
            self._block_dict[self._blocks_name[id]] = block

    def startClient(self):
        def getblock(i):
            return self._block_dict[self._blocks_name[block_id[i]]]

        while True:
            try:
                self._createClient()
                self.conn.connect((self.server_ip, self.server_port))
                break
            except ConnectionRefusedError:
                # print('重试连接中...')
                pass
            time.sleep(1)

        (block_id, sps), _ = get_data(self.conn)
        block_num = len(block_id)

        self._getblocks(block_id, sps)

        print(f'边缘设备选择的块：{[self._blocks_name[id] for id in block_id]}')
        print(f'对应的sparsity：{sps}')

        is_gpu = self._device == 'cuda'
        edge_latency = 0.
        down_transmit_latency = 0.

        get_message(self.conn, show=False)

        if block_num != 0:
            if block_id[0] != 0:
                send_data(self.conn, self._x, '输入数据')

                send_message(self.conn, 'break', show=False)

                self._x, t = get_data(self.conn)
                if self._x.device != self._device:
                    self._x = self._x.to(self._device)

                down_transmit_latency += t

                get_message(self.conn, show=False)
            print(f'处理{self._blocks_name[block_id[0]]}中')
            self._x, t = Inference(getblock(0), self._x, is_gpu)
            edge_latency += t

            for i in range(1, block_num):
                if block_id[i] - 1 != block_id[i - 1]:
                    send_data(self.conn, self._x, 'IR')

                    send_message(self.conn, 'break', show=False)

                    self._x, down_l = get_data(self.conn)
                    if self._x.device != self._device:
                        self._x = self._x.to(self._device)

                    down_transmit_latency += down_l

                    get_message(self.conn, show=False)

                print(f'处理{self._blocks_name[block_id[i]]}中')
                self._x, t = Inference(getblock(i), self._x, is_gpu)
                edge_latency += t

            if self._blocks_name[block_id[-1]] != self._blocks_name[-1]:
                send_data(self.conn, self._x, 'IR')

                send_message(self.conn, 'Get result', show=False)

                pack, down_l = get_data(self.conn)

                down_transmit_latency += down_l

                cloud_latency, up_transmit_latency, res = pack

            else:
                send_message(self.conn, 'Get result', show=False)

                pack, _ = get_data(self.conn)

                cloud_latency, up_transmit_latency = pack
                res = self._x

        else:
            send_data(self.conn, self._x, '输入数据')

            send_message(self.conn, 'Get result', show=False)

            pack, t = get_data(self.conn)

            down_transmit_latency += t

            cloud_latency, up_transmit_latency, res = pack
            res = self._x

        self._res['cloud_infer_latency'] = cloud_latency
        self._res['edge_infer_latency'] = edge_latency
        self._res['upload_transmit_latency'] = up_transmit_latency
        self._res['download_transmit_latency'] = down_transmit_latency
        self._res['all_latency'] = cloud_latency + edge_latency + up_transmit_latency + down_transmit_latency
        self._res['result'] = res

        close_conn(self.conn)

# For multi-devices
class MainClient(Client):
    def __init__(self, ip, port, dict):
        super().__init__(ip, port)
        self._createClient()
        self.di = dict

    def startClient(self, *args):
        while True:
            try:
                self.conn.connect((self.server_ip, self.server_port))
                self.di['client'] = self.conn
                logger.info('连接服务器成功!')
                break
            except ConnectionError:
                pass
            time.sleep(1)

class ProfilesRecver(Process):
    def __init__(self, _trained_path, dict):
        super().__init__()
        self._conn = dict['client']
        self._trained_path = _trained_path

    def _checkMetrics(self):
        li = os.listdir(self._trained_path)
        corr = 0
        for n in li:
            if ('server' in n and n.endswith('.csv')) or \
                    ('server' in n and n.endswith('.yaml')):
                corr += 1
        return corr >= 2

    def run(self):
        if not os.path.exists(self._trained_path) or not self._checkMetrics():
            logger.info('开始下载训练数据')
            logger.info(self._trained_path)
            create_dir(self._trained_path)
            send_message(self._conn, 'Block Request', msg='请求下载已训练的块', show=True)
            block_nums = get_short_data(self._conn)

            send_message(self._conn, 'break', 'break', False)

            for _ in range(block_nums + 1):
                tmp, _ = get_data(self._conn)
                block, block_name = tmp
                save_model(block, os.path.join(self._trained_path, block_name), ModelSaveMethod.FULL)

            tmp, _ = get_data(self._conn)
            csv, csv_name = tmp
            csv.to_csv(os.path.join(self._trained_path, csv_name), index=False)

            tmp, _ = get_data(self._conn)
            y, yaml_name = tmp
            with open(os.path.join(self._trained_path, yaml_name), 'w') as f:
                yaml.dump(y, f)
            logger.info('成功获得数据')
        else:
            logger.info('无需接收数据')
            send_message(self._conn, 'Fine', msg='无需下载', show=False)

class ProfilesChecker(Process):
    def __init__(self, offload_path, dict):
        super().__init__()
        self.conn = dict['client']
        self._offload_path = offload_path

    def run(self):
        s = get_message(self.conn, show=False)

        if s == 'Require edge profiles':
            logger.info('开始上传信息')

            df = pd.read_csv(os.path.join(self._offload_path, 'edge-blocks-metrics.csv'))
            pack = [df, 'edge-blocks-metrics.csv']
            send_data(self.conn, pack, 'Edge Metrics', True)

            f = open(os.path.join(self._offload_path, 'edge-teacher-model-metrics.yaml'))
            metrics = yaml.load(f, yaml.Loader)
            pack = [metrics, 'edge-teacher-model-metrics.yaml']
            send_data(self.conn, pack, 'edge-teacher-model-metrics.yaml', True)

            logger.info('成功上传数据')
        else:
            logger.info('无需上传数据')

class MultiMoniterBWupClient:
    def __init__(self, dict):
        super().__init__()
        self.conn = dict['client']

    def start(self):
        x = torch.rand((1, 3, 224, 224)).cpu()
        send_data(self.conn, x)

class MultiMoniterBWdownClient:
    def __init__(self, dict):
        super().__init__()
        self.conn = dict['client']

    def start(self):
        bw = get_bandwidth(self.conn)

        get_message(self.conn, show=False)  # 防止粘包

        send_short_data(self.conn, bw, 'Bandwidth', show=False)

class MultiMoniterDeviceInfoClient:
    def __init__(self, dict, core_num):
        super().__init__()
        self.conn = dict['client']
        self.core_num = core_num

    def start(self):
        os.sched_setaffinity(os.getpid(), list(range(self.core_num)))
        l_th, m_th, flops = monitorCPUInfo()

        # logger.info(f'{l_th}, {m_th}')
        send_short_data(self.conn, (l_th, m_th, flops), msg='device info', show=False)

        get_message(self.conn, show=False)

class MultiDeployClient:
    def __init__(self, block_manager, trained_path, device, dict):
        super().__init__()
        self._block_manager = block_manager
        self._trained_path = trained_path
        self._blocks_name = None
        self._block_dict = None
        self._device = device
        self.conn = dict['client']
        self.task_queue = []

    # def _getblocks(self, block_id, sps):
    #     self._blocks_name = self._block_manager.get_blocks_id()
    #     self._block_dict = {}
    #     for id, sp in zip(block_id, sps):
    #         block = self._block_manager.get_block_from_file(
    #             os.path.join(self._trained_path,self._block_manager.get_block_file_name(self._blocks_name[id], sp)),
    #             self._device).to(self._device)
    #         self._block_dict[self._blocks_name[id]] = block

    @abc.abstractmethod
    def _getblocks(self, block_id, sps):
        pass

    @abc.abstractmethod
    def start(self):
        pass

    def _processDeployInfos(self):
        task_queue = []

        (block_id, sps), _ = get_data(self.conn)

        self._getblocks(block_id, sps)

        tmp_model = []
        self._layers = []
        pre_id = -2
        # 1: Inference
        # 2: Upload to server
        # 3: Download from server
        for i, id in enumerate(block_id):
            # print(pre_id)
            if pre_id + 1 == id:
                tmp_model.append(self._block_dict[self._blocks_name[id]])
            else:
                if len(tmp_model) > 0:
                    tmp_model = nn.Sequential(*tmp_model)
                    self._layers.append(tmp_model)
                    task_queue.append((1, len(self._layers) - 1))
                    task_queue.append((2, None))
                task_queue.append((3, None))
                tmp_model = [self._block_dict[self._blocks_name[id]]]
            pre_id = id
        if len(tmp_model) > 0:
            tmp_model = nn.Sequential(*tmp_model)
            self._layers.append(tmp_model)
            task_queue.append((1, len(self._layers) - 1))
            task_queue.append((2, None))

        return task_queue

    def startClient(self):
        is_gpu = self._device == 'cuda'
        edge_latency = 0.
        down_transmit_latency = 0.
        flag = get_message(self.conn, False)
        download_size = 0.
        send_message(self.conn, 'break')

        if flag == 'Changed':
            self.task_queue = self._processDeployInfos()
        # print(flag)

        for flag, arg in self.task_queue:
            # print((flag, arg))
            if flag == 1:
                layer = self._layers[arg]
                self._x, latency = Inference(layer, self._x, is_gpu)
                edge_latency += latency
            elif flag == 2:
                get_message(self.conn, False)
                send_data(self.conn, self._x)
                send_message(self.conn, 'break')
            elif flag == 3:
                self._x, latency = get_data(self.conn)
                down_transmit_latency += latency
                download_size += torch.numel(self._x) * 4 / (1024 ** 2)

        get_message(self.conn, False)
        send_short_data(self.conn, edge_latency, 'edge latency', False)
        get_message(self.conn, False)
        send_short_data(self.conn, down_transmit_latency, 'download latency', False)
        get_message(self.conn, False)
        bwdown = download_size / ((down_transmit_latency + 1e-5) / 1000)
        send_short_data(self.conn, bwdown, 'real download bw', False)
        # print(f'{download_size}MB, {down_transmit_latency}ms, {bwdown}MB/s')
        flag = get_message(self.conn, False)
        if flag == 'Check':
            send_message(self.conn, 'break')
            x, latency = get_data(self.conn)
            bwdown = (torch.numel(x) * 4 / (1024 ** 2)) / ((latency + 1e-5) / 1000)
            get_message(self.conn, False)
            send_data(self.conn, (bwdown, x))
            send_message(self.conn, 'break')
        else:
            send_message(self.conn, 'break')
