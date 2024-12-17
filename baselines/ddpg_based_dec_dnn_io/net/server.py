import abc
import os
import json
import torch
from multiprocessing import Process
from net.util import *
import platform
from util.model.util import warmUP
from branchynet.util import getEarlyExitChain
from util.model.util import getPartitionedModels, Inference
import random
from tqdm import tqdm

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

class MainServer(Server):
    def __init__(self, ip, port, dict, num_clients=2):
        super().__init__(ip, port)
        self._createServer(max_client_num=num_clients)

        self.di = dict

    def startServer(self):
        li = []
        while len(li) != self.max_client_num:
            conn, client = self.p.accept()
            print(f'客户端-{len(li)}已经成功连接')
            li.append(conn)

        self.di['server'] = self.p
        self.di['clients'] = li

class TranmitModelServer:
    def __init__(self, device_dict, path):
        self.server = device_dict['server']
        self.clients = device_dict['clients']
        self.path = path

    def start(self):
        for id, conn in enumerate(self.clients):
            send_message(conn, 'model')

            flag = get_message(conn, False)
            if flag == 'request':
                print(f'传输模型至客户端-{id}')
                send_data(conn, torch.load(self.path, map_location='cpu'))
                print(f'传输完成')
            else:
                print(f'客户端-{id}模型已存在')

class DefaultDeployServer:
    def __init__(self, device_dict, path, model, deploy_info, test_loader, update_flag, running_flag, device):
        self.server = device_dict['server']
        self.clients = device_dict['clients']
        self.dataloader = test_loader
        self.deploy_info = deploy_info
        self.model = model
        self.device = device
        self.update_flag = update_flag
        self.running_flag = running_flag
        self.device_list = []
        self.s = []
        self.models = []
        self.path = path
    def _processDeployInfos(self):
        infos = []
        for i in range(len(self.clients)):
            infos.append([self.deploy_info['info'][j][i] for j in range(len(self.deploy_info['info']) - 1)])
        F = self.deploy_info['info'][2]
        if self.device == 'cpu':
            start_cpu_id = 0
            for fi in F:
                self.device_list.append(list(range(start_cpu_id, start_cpu_id + fi)))
                start_cpu_id += fi
        return infos

    def start(self):
        sum_latency = 0.
        sum_corr = 0.
        test_num = 0.
        length = len(self.dataloader)
        pbar = tqdm(total=length)
        dataloader = iter(self.dataloader)
        id = 0
        deploy_changed = True
        sign = ['Changed' for i in range(len(self.clients))]
        infos = None
        mem_array = []
        l_array = []
        flops_array = []
        acc_array = []
        while id < length:
            if self.update_flag.value == 1:
                self.running_flag.value = 0
                x, y = next(dataloader)
                pbar.set_description(f'Testing task-{id}...')

                if deploy_changed:
                    infos = self._processDeployInfos()
                    self.s = []
                    self.models = []
                    for conn_id in range(len(self.clients)):
                        model = getEarlyExitChain(self.model, infos[conn_id][1])
                        _, model = getPartitionedModels(model, infos[conn_id][0])
                        self.models.append(model)
                        sign[conn_id] = 'Changed'
                    deploy_changed = False

                    # warmUP(self.model, x, repeat=10)
                corr, latency, mem, flops = self._startServer(x, y, infos, sign)
                test_num += x.shape[0]
                sum_corr += corr
                sum_latency += latency
                id += 1
                pbar.update(1)
                pbar.set_postfix(avg_acc=sum_corr / test_num, mean_latency=sum_latency / id)
                self.running_flag.value = 1
                mem_array.append(mem / (1024 ** 2))
                flops_array.append(flops / 1e6)
                l_array.append(sum_latency / id)
                acc_array.append(sum_corr / test_num)
                time.sleep(0.005)
            else:
                deploy_changed = True

        json.dump([acc_array, l_array, mem_array, flops_array], open(os.path.join(self.path, 'ddpg_based_dec_dnn_io.json'), 'w'))

        for i, conn in enumerate(self.clients):
            send_message(conn, 'Stop')


    def _startServer(self, x, y, infos, sign):
        isgpu = True if self.device != 'cpu' else False

        conn_id = random.choice(list(range(len(self.clients))))
        os.sched_setaffinity(os.getpid(), self.device_list[conn_id])
        conn = self.clients[conn_id]
        cloud_latency = 0.
        send_message(conn, sign[conn_id])

        get_message(conn, False)

        if sign[conn_id] == 'Changed':
            send_data(conn, infos[conn_id])
            sign[conn_id] = 'Start'


        model = self.models[conn_id]

        send_data(conn, x)

        send_message(conn, 'break')

        tmp, up_time = get_data(conn)

        get_message(conn, False)

        ir, edge_latency = tmp
        if ir.device != self.device:
            ir = ir.to(self.device)

        for layer in model:
            ir, latency = Inference(layer, ir, isgpu)
            cloud_latency += latency

        latency = edge_latency + up_time + cloud_latency
        print(f'edge_latency={edge_latency}, up_time={up_time}, cloud_latency={cloud_latency}')
        pred = torch.argmax(ir, dim=1)
        corr = torch.sum(pred == y).item()

        return corr, latency, self.deploy_info['memory'][conn_id], self.deploy_info['FLOPs'][conn_id]

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
    def __init__(self, dict, cpu_cores, f_array):
        super().__init__()
        self.p = dict['server']
        self.conns = dict['clients']
        self.f_array = f_array
        self.cpu_cores = cpu_cores


    def start(self):
        l_th, m_th, flops = moniterServerCPUInfo(self.cpu_cores)
        self.f_array[0] = flops
        # logger.info('success')
        for i, conn in enumerate(self.conns):
            tmp = get_short_data(conn)
            l_t, m_t, flops = tmp
            self.f_array[i + 1] = flops

            send_message(conn, 'break')