import abc
import os.path
from multiprocessing import Process
import platform
import torch.nn as nn
import yaml

from offload.inference_utils import monitorCPUInfo
from offload.net.utils import *
from offload.deployment import divideModel
from offload.inference_utils import getModel, warmUP, Inference
import pandas as pd
from legodnn.utils.common.log import logger
from legodnn.utils.common.file import create_dir
from collections import deque
from copy import deepcopy

class Server(Process):
    def __init__(self, ip, port):
        super().__init__()
        self.ip = ip
        self.port = port
        self.p = None
        self.conn = None
        self.max_client_num = None

    def _createServer(self, max_client_num=10):
        self.p = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.max_client_num = max_client_num

        # 判断使用的是什么平台
        sys_platform = platform.platform().lower()
        if "windows" in sys_platform:
            self.p.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # windows
        else:
            self.p.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # macos or linux

        self.p.bind((self.ip, self.port))  # 绑定端口号
        self.p.listen(max_client_num)  # 打开监听


    @abc.abstractmethod
    def startServer(self):
        pass

    def run(self) -> None:
        self.startServer()

#For 1 cloud 1 edge

class LegodnnProfileServer(Server):
    def __init__(self, ip, port, block_manager, trained_path):
        super().__init__(ip, port)
        self._block_manager = block_manager
        self._trained_path = trained_path

    def startServer(self):
        self._createServer()

        self.conn, client = self.p.accept()

        msg = get_message(self.conn)

        if msg == 'Block Request':
            logger.info('开始传输')
            sparsities = self._block_manager.get_blocks_sparsity()
            block_ids = self._block_manager.get_blocks_id()
            block_num = len(block_ids)

            send_short_data(self.conn, block_num, True)
            get_message(self.conn, False)

            for id in range(block_num):
                sp_list = sparsities[id] + [0.]
                for sp in sp_list:
                    b_name = self._block_manager.get_block_file_name(id, sp)
                    path = os.path.join(self._trained_path, b_name)
                    block = self._block_manager.get_block_from_file(path, 'cpu')
                    pack = [block, b_name]
                    send_data(self.conn, pack, 'Block', show=True)

                df = pd.read_csv(os.path.join(self._trained_path, 'server-blocks-metrics.csv'))
                pack = [df, 'server-blocks-metrics.csv']
                send_data(self.conn, pack, 'server-blocks-metrics.csv', True)

                f = open(os.path.join(self._trained_path, 'server-teacher-model-metrics.yaml'))
                metrics = yaml.load(f, yaml.Loader)
                pack = [metrics, 'server-teacher-model-metrics.yaml']
                send_data(self.conn, pack, 'server-teacher-model-metrics.yaml', True)
            logger.info('传递成功')
        else:
            logger.info('无需传递')

        close_conn(self.conn)
        close_socket(self.p)


class LegodnnRecvingServer(Server):
    def __init__(self, ip, port, offload_path):
        super().__init__(ip, port)
        self._offload_path = offload_path

    def startServer(self):
        self._createServer()

        self.conn, client = self.p.accept()

        if not os.path.exists(self._offload_path):
            logger.info('开始接收')
            send_message(self.conn, 'Require edge profiles', True)
            create_dir(self._offload_path)

            csv, csv_name = get_data(self.conn)
            csv.to_csv(os.path.join(self._offload_path, csv_name), index=False)

            y, yaml_name = get_data(self.conn)
            with open(os.path.join(self._offload_path, yaml_name), 'w') as f:
                yaml.dump(y, f)

            logger.info('传递成功')
        else:
            send_message(self.conn, 'Fine', True)
            logger.info('无需传递')

        close_conn(self.conn)
        close_socket(self.p)

class DefaultDeployServer(Server):
    def __init__(self, ip, port, is_gpu=True):
        super().__init__(ip, port)
        self.is_gpu = is_gpu

    def startServer(self):
        self._createServer()

        self.conn, client = self.p.accept()

        model_type = get_short_data(self.conn)
        print(f'得到模型种类为{model_type}')

        model = getModel(model_type)
        point = get_short_data(self.conn)
        print(f'得到模型断点为第{point}层')

        if len(model) != point:
            self.conn.sendall('break'.encode())
            print('获取IR中...')
            x, trans_latency = get_data(self.conn)

            _, cloud_model = divideModel(model, point)
            cloud_model = cloud_model.to('cuda' if self.is_gpu else 'cpu')

            x = x.to('cuda' if self.is_gpu else 'cpu')
            print(f'传输时间为第{trans_latency}ms')

            self.conn.recv(40)

            send_short_data(self.conn, trans_latency, 'Trans latency')

            warmUP(cloud_model, x)
            x, latency = Inference(cloud_model, x, self.is_gpu)

            self.conn.recv(40)

            send_short_data(self.conn, latency, 'Cloud latency')

            self.conn.recv(40)

            send_data(self.conn, x, 'Result')

        close_conn(self.conn)
        close_socket(self.p)



class MoniterServer(Server):
    def __init__(self, ip, port):
        super().__init__(ip, port)

    def startServer(self):
        self._createServer()

        self.conn, client = self.p.accept()

        bw = get_bandwidth(self.conn)

        get_short_data(self.conn)  # 防止粘包

        send_short_data(self.conn, bw, 'Bandwidth', show=False)

        close_conn(self.conn)
        close_socket(self.p)

class MoniterBWdowmServer(Server):
    def __init__(self, ip, port, bw, first=False):
        super().__init__(ip, port)
        self.bw = bw
        self.first = first

    def startServer(self):
        self._createServer()
        if not self.first:
            self.p.settimeout(2)

        try:
            self.conn, client = self.p.accept()

            x = torch.rand((1, 3, 224, 224))
            gamma = 0.5

            send_data(self.conn, x)

            send_message(self.conn, 'break', show=False)

            if self.bw.value != 0:
                bw = gamma * get_short_data(self.conn) + (1 - gamma) * self.bw.value
                # bw = min(self.bw.value, get_short_data(self.conn))
            else:
                bw = get_short_data(self.conn)

            self.bw.value = bw
            close_conn(self.conn)
        except Exception:
            pass

        close_socket(self.p)

class MoniterBWupServer(Server):
    def __init__(self, ip, port, bw, first=False):
        super().__init__(ip, port)
        self.bw = bw
        self.first = first

    def startServer(self):
        gamma = 0.5
        self._createServer()
        if not self.first:
            self.p.settimeout(2)

        try:
            self.conn, client = self.p.accept()

            if self.bw.value != 0:
                self.bw.value = gamma * get_bandwidth(self.conn) + (1 - gamma) * self.bw.value
                # bw = min(self.bw.value, get_short_data(self.conn))
            else:
                self.bw.value = get_bandwidth(self.conn)

            close_conn(self.conn)
        except Exception:
            pass

        close_socket(self.p)

class LegodnnDeployServer(Server):
    def __init__(self, ip, port, deploy_infos, block_manager, trained_path, device):
        super().__init__(ip, port)
        self._block_manager = block_manager
        self._trained_path = trained_path
        self._device = device
        self._deploy_infos = deploy_infos
        self._x = None

    def _getblocks(self, block_id, sps):
        self._blocks_name = self._block_manager.get_blocks_id()
        self._block_dict = {}
        for id, sp in zip(block_id, sps):
            block = self._block_manager.get_block_from_file(
                os.path.join(self._trained_path, self._block_manager.get_block_file_name(self._blocks_name[id], sp)),
                self._device).to(self._device)
            self._block_dict[self._blocks_name[id]] = block

    def _processDeployInfos(self):
        sps = self._deploy_infos['blocks_sparsity']
        dvs = self._deploy_infos['blocks_devices']

        cloud_sps = []
        cloud_blocks = []
        edge_sps = []
        edge_blocks = []

        for i, (sp, dv) in enumerate(zip(sps, dvs)):
            if dv == 'cloud':
                cloud_sps.append(sp)
                cloud_blocks.append(i)
            else:
                edge_sps.append(sp)
                edge_blocks.append(i)

        return cloud_sps, cloud_blocks, edge_sps, edge_blocks

    def startServer(self):
        def getblock(i):
            return self._block_dict[self._blocks_name[block_id[i]]]

        self._createServer()
        self.conn, client = self.p.accept()

        sps, block_id, edge_sps, edge_blocks = self._processDeployInfos()
        block_num = len(block_id)

        is_gpu = self._device == 'cuda'
        cloud_latency = 0.
        up_transmit_latency = 0.
        self._getblocks(block_id, sps)

        print(f'云服务器设备选择的块：{[self._blocks_name[id] for id in block_id]}')
        print(f'对应的sparsity：{sps}')

        send_data(self.conn, (edge_blocks, edge_sps), '部署信息', show=False)

        send_message(self.conn, 'break', show=False)

        #
        if len(block_id) != 0:
            self._x, t = get_data(self.conn)
            if self._x.device != self._device:
                self._x = self._x.to(self._device)

            up_transmit_latency += t

            get_message(self.conn, show=False)

            print(f'处理{self._blocks_name[block_id[0]]}中')
            self._x, t = Inference(getblock(0), self._x, is_gpu)

            cloud_latency += t

            for i in range(1, block_num):
                if block_id[i] - 1 != block_id[i - 1]:
                    send_data(self.conn, self._x, 'IR')

                    send_message(self.conn, 'break', show=False)

                    self._x, t = get_data(self.conn)
                    if self._x.device != self._device:
                        self._x = self._x.to(self._device)

                    up_transmit_latency += t

                    get_message(self.conn, show=False)

                print(f'处理{self._blocks_name[block_id[i]]}中')
                self._x, t = Inference(getblock(i), self._x, is_gpu)

                cloud_latency += t

            if self._blocks_name[block_id[-1]] != self._blocks_name[-1]:
                send_data(self.conn, self._x, 'IR')

                send_message(self.conn, 'break', show=False)

                get_message(self.conn, show=False) # Get result

                send_data(self.conn, (cloud_latency, up_transmit_latency), '推理信息', show=False)

            else:
                send_data(self.conn, (cloud_latency, up_transmit_latency, self._x), '推理信息', show=False)

        else:
            get_message(self.conn, show=False) # Get result

            send_data(self.conn, (cloud_latency, up_transmit_latency), '推理信息', show=False)

        close_conn(self.conn)
        close_socket(self.p)

# For multi-devices
class MainServer(Server):
    def __init__(self, ip, port, dict, num_clients=2):
        super().__init__(ip, port)
        self._createServer(max_client_num=num_clients)

        self.di = dict

    def startServer(self):
        li = []
        while len(li) != self.max_client_num:
            conn, client = self.p.accept()
            logger.info(f'客户端-{len(li)}已经成功连接')
            li.append(conn)

        self.di['server'] = self.p
        self.di['clients'] = li


class ProfilesChecker(Process):
    def __init__(self, trained_path, dict, block_manager):
        super().__init__()
        self._server = dict['server']
        self._clients = dict['clients']
        self._block_manager = block_manager
        self._trained_path = trained_path

    def run(self):
        for i, conn in enumerate(self._clients):
            logger.info(f'检查客户端-{i}是否存在模型信息')
            msg = get_message(conn, show=False)

            if msg == 'Block Request':
                logger.info(f'客户端-{i}不存在相关信息，开始传输')
                sparsities = self._block_manager.get_blocks_sparsity()
                block_ids = self._block_manager.get_blocks_id()
                block_num = sum(map(lambda x:len(x), sparsities))

                send_short_data(conn, block_num, '块数量信息', True)
                get_message(conn, False)

                for id in range(len(block_ids)):
                    sp_list = sparsities[id]
                    for sp in sp_list:
                        b_name = 'block-' + self._block_manager.get_block_file_name(id, sp)
                        path = os.path.join(self._trained_path, b_name)
                        block = self._block_manager.get_block_from_file(path, 'cpu')
                        pack = [block, b_name]
                        send_data(conn, pack, 'Block', show=True)

                model_frame = torch.load(os.path.join(self._trained_path, 'model_frame.pt'), map_location='cpu')
                pack = [model_frame, 'model_frame.pt']
                send_data(conn, pack, 'Block', show=True)

                df = pd.read_csv(os.path.join(self._trained_path, 'server-blocks-metrics.csv'))
                pack = [df, 'server-blocks-metrics.csv']
                send_data(conn, pack, 'server-blocks-metrics.csv', True)

                f = open(os.path.join(self._trained_path, 'server-teacher-model-metrics.yaml'))
                metrics = yaml.load(f, yaml.Loader)
                pack = [metrics, 'server-teacher-model-metrics.yaml']
                send_data(conn, pack, 'server-teacher-model-metrics.yaml', True)
                logger.info('传递成功')
            else:
                logger.info(f'客户端-{i}存在模型信息，无需传递')


class ProfilesRecver(Process):
    def __init__(self, trained_path, offload_path, dict, block_manager):
        super().__init__()
        self._server = dict['server']
        self._clients = dict['clients']
        self._block_manager = block_manager
        self._trained_path = trained_path
        self._offload_path = offload_path

    def _checkMetrics(self, i):
        li = os.listdir(self._offload_path)
        corr = 0
        for n in li:
            if ('edge' in n and n.endswith(f'_{i}.csv')) or \
                    ('edge' in n and n.endswith(f'_{i}.yaml')):
                corr += 1
        return corr == 2

    def run(self):
        for i, conn in enumerate(self._clients):
            logger.info(f'检查是否需要接收客户端-{i}的模型运行信息')
            # ed = '_'.join(conn.getpeername())
            ed = conn.getpeername()[0]
            if not os.path.exists(self._offload_path) or not self._checkMetrics(ed):
                logger.info('开始接收')
                send_message(conn, 'Require edge profiles', '传递边缘块延迟信息', True)
                create_dir(self._offload_path)

                tmp, _ = get_data(conn)
                csv, csv_name = tmp
                csv_name = f'{csv_name.split(".")[0]}_{ed}.csv'
                csv.to_csv(os.path.join(self._offload_path, csv_name), index=False)

                tmp, _ = get_data(conn)
                y, yaml_name = tmp
                yaml_name = f'{yaml_name.split(".")[0]}_{ed}.yaml'
                with open(os.path.join(self._offload_path, yaml_name), 'w') as f:
                    yaml.dump(y, f)
                logger.info('接收成功')
            else:
                send_message(conn, 'Fine', '', False)
                logger.info('无需接收')

class MultiMoniterBWdownServer:
    def __init__(self, dict, bws):
        super().__init__()
        self.p = dict['server']
        self.conns = dict['clients']
        self.bws = bws

    def start(self):
        x = torch.rand((1, 3, 224, 224)).cpu()
        gamma = 0.5
        for i, conn in enumerate(self.conns):
            send_data(conn, x)

            send_message(conn, 'break', show=False)

            if self.bws[i] != 0:
                bw = gamma * get_short_data(conn) + (1 - gamma) * self.bws[i]
                # bw = min(self.bw.value, get_short_data(conn))
            else:
                bw = get_short_data(conn)

            self.bws[i] = bw

class MultiMoniterBWupServer:
    def __init__(self, dict, bws):
        super().__init__()
        self.p = dict['server']
        self.conns = dict['clients']
        self.bws = bws

    def start(self):
        gamma = 0.5

        for i, conn in enumerate(self.conns):
            bw = self.bws[i]
            if bw != 0:
                bw = gamma * get_bandwidth(conn) + (1 - gamma) * bw
                # bw = min(self.bw.value, get_short_data(self.conn))
            else:
                bw = get_bandwidth(conn)
            self.bws[i] = bw


class MultiMoniterDeviceInfoServer:
    def __init__(self, dict, l_th, m_th, f_array):
        super().__init__()
        self.p = dict['server']
        self.conns = dict['clients']
        self.l_th = l_th
        self.m_th = m_th
        self.f_array = f_array


    def start(self):
        l_th, m_th, flops = monitorCPUInfo()
        self.l_th[0] = l_th / 20
        self.m_th[0] = m_th / 20
        self.f_array[0] = flops

        for i, conn in enumerate(self.conns):
            tmp = get_short_data(conn)
            l_t, m_t, flops = tmp
            self.l_th[i + 1] = l_t / 20
            self.m_th[i + 1] = m_t / 20
            self.f_array[i + 1] = flops

            send_message(conn, 'break')

class MultiDeployServer:
    def __init__(self, datasets, model_frame, deploy_info, block_manager, trained_path, device, dict, bwdown, bwup):
        super().__init__()
        self._block_manager = block_manager
        self._trained_path = trained_path
        self._device = device
        self._deploy_infos = deploy_info
        self.p = dict['server']
        self.conns = dict['clients']
        self.edge_num = len(self.conns)
        self._root, self._end, self._frame = model_frame
        self._len_dataset = len(datasets)
        self._datasets = iter(datasets)
        self._x = None
        self._layers = []
        self._res = {}
        self._task_queue = None
        self._block_dict = {}
        self._blocks_name = []
        self._bwdown = bwdown
        self._bwup = bwup


    # def _getblocks(self, block_id, sps):
    #     self._blocks_name = self._block_manager.get_blocks_id()
    #     self._block_dict = {}
    #     for id, sp in zip(block_id, sps):
    #         block = self._block_manager.get_block_from_file(
    #             os.path.join(self._trained_path, self._block_manager.get_block_file_name(self._blocks_name[id], sp)),
    #             self._device).to(self._device)
    #         self._block_dict[self._blocks_name[id]] = block

    @abc.abstractmethod
    def _getblocks(self, block_id, sps):
        pass

    def _processDeployInfos(self):
        sps = self._deploy_infos['infos']['readable_sparsity']
        dvs = self._deploy_infos['infos']['blocks_devices']
        self._res['FLOPs'] = self._deploy_infos['infos']['FLOPs']
        self._res['model_size'] = self._deploy_infos['infos']['model_size']

        cloud_sps = []
        cloud_blocks = []
        task_queue = deque()

        edge_sps = [list() for _ in range(self.edge_num)]
        edge_blocks = [list() for _ in range(self.edge_num)]

        self.last_block = len(dvs) - 1

        for i, (sp, dv) in enumerate(zip(sps, dvs)):
            if dv == 0:
                cloud_sps.append(sp)
                cloud_blocks.append(i)
            else:
                edge_sps[dv - 1].append(sp)
                edge_blocks[dv - 1].append(i)

        self._getblocks(cloud_blocks, cloud_sps)

        tmp_model = [self._root]
        self._layers = []
        pre_dv = -1
        # 1: Inference
        # 2: Upload from client-x
        # 3: Download to client-x
        for i, dv in enumerate(dvs):
            if dv == 0:
                if pre_dv != -1 and pre_dv != 0:
                    task_queue.append((2, pre_dv - 1))
                tmp_model.append(self._block_dict[self._blocks_name[i]])
                if i == len(dvs) - 1:
                    tmp_model.append(self._end)
                    tmp_model = nn.Sequential(*tmp_model)
                    self._layers.append(tmp_model)
                    task_queue.append((1, len(self._layers) - 1))
            else:
                if i == 0:
                    tmp_model = nn.Sequential(*tmp_model)
                    self._layers.append(tmp_model)
                    tmp_model = []
                    task_queue.append((1, len(self._layers) - 1))
                    task_queue.append((3, dv - 1))
                if pre_dv == 0:
                    tmp_model = nn.Sequential(*tmp_model)
                    self._layers.append(tmp_model)
                    tmp_model = []
                    task_queue.append((1, len(self._layers) - 1))
                    task_queue.append((3, dv - 1))
                elif pre_dv != dv and pre_dv != -1:
                    task_queue.append((2, pre_dv - 1))
                    task_queue.append((3, dv - 1))
                if i == len(dvs) - 1:
                    task_queue.append((2, dv - 1))
                    self._layers.append(self._end)
                    task_queue.append((1, len(self._layers) - 1))
            pre_dv = dv
        return edge_sps, edge_blocks, task_queue

    @abc.abstractmethod
    def start(self):
        pass

    def _startServer(self, sign):
        is_gpu = self._device == 'cuda'
        cloud_latency = 0.
        edge_latency = 0.
        up_transmit_latency = [0. for _ in range(len(self.conns))]
        down_transmit_latency = 0.
        upload_size = [0. for _ in range(len(self.conns))]

        if sign == 'Changed':
            edge_sps, edge_blocks, self.task_queue = self._processDeployInfos()
            # logger.info(f'服务端设备选择的块：{[self._blocks_name[id] for id in block_id]}')
            # logger.info(f'对应的sparsity：{sps}')

            for i, conn in enumerate(self.conns):
                send_message(conn, sign)

                get_message(conn, False)

                send_data(conn, (edge_blocks[i], edge_sps[i]), '部署信息', show=False)

        else:
            for i, conn in enumerate(self.conns):
                send_message(conn, sign)

                get_message(conn, False)

        new_queue = deque()
        while len(self.task_queue) != 0:
            flag, arg = self.task_queue.popleft()
            # print((flag, arg))
            new_queue.append((flag, arg))
            if flag == 1:
                layer = self._layers[arg]
                self._x, latency = Inference(layer, self._x, is_gpu)
                cloud_latency += latency
            elif flag == 2:
                send_message(self.conns[arg], 'break')
                self._x, latency = get_data(self.conns[arg])
                get_message(self.conns[arg], False)
                up_transmit_latency[arg] += latency
                upload_size[arg] += torch.numel(self._x) * 4 / (1024 ** 2)
            elif flag == 3:
                send_data(self.conns[arg], self._x)

        self.task_queue = new_queue

        for i, conn in enumerate(self.conns):
            send_message(conn, 'get latency')
            edge_latency += get_short_data(conn)
            send_message(conn, 'break')
            down_transmit_latency += get_short_data(conn)
            send_message(conn, 'break')
            now_bwdown = get_short_data(conn)
            now_bwup = upload_size[i] / ((up_transmit_latency[i] + 1e-5) / 1000)
            bw_check = False

            if now_bwdown != 0:
                self._bwdown[i] = now_bwdown
                bw_check = True
            if now_bwup != 0:
                self._bwup[i] = now_bwup
                bw_check = True

            if bw_check:
                send_message(conn, 'Check')
                get_message(conn, False)
                x = torch.rand((1, 3, 32, 32))
                send_data(conn, x)
                send_message(conn, 'break')
                (self._bwdown[i], x), latency = get_data(conn)
                self._bwup[i] = (torch.numel(x) * 4 / (1024 ** 2)) / ((latency + 1e-5) / 1000)
                get_message(conn, False)
            else:
                send_message(conn, 'No')
                get_message(conn, False)

        up_transmit_latency = sum(up_transmit_latency)

        self._res['output'] = self._x
        self._res['cloud_latency'] = cloud_latency
        self._res['edge_latency'] = edge_latency
        self._res['up_transmit_latency'] = up_transmit_latency
        self._res['down_transmit_latency'] = down_transmit_latency
        self._res['total_latency'] = cloud_latency + edge_latency + up_transmit_latency + down_transmit_latency

def getClientNames(dict):
    return [conn.getpeername()[0] for conn in dict['clients']]

def closeDevices(dict):
    for conn in dict['clients']:
        close_conn(conn)

    close_socket(dict['server'])