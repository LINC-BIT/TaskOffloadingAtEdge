import abc
from multiprocessing import Process
from util.data.log import logger
from net.util import *
from branchynet.util import getEarlyExitChain
from util.model.util import getPartitionedModels, Inference
import os
import torch
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


class TranmitModelClient:
    def __init__(self, device_dict, path):
        self.conn = device_dict['client']
        self.path = path

    def start(self):
        get_message(self.conn, False)

        if not os.path.exists(self.path):
            send_message(self.conn, 'request')
            print('开始接收模型')
            model, _ = get_data(self.conn)
            print('开始接收完成')
            torch.save(model, self.path)
        else:
            print('无需接收')
            send_message(self.conn, 'no')

class DefaultClient:
    def __init__(self, client_dict, model, used_cpu, device):
        self.conn = client_dict['client']
        self.model = model
        self.edge_model = None
        self.cpu_list = list(range(used_cpu))
        self.device = device

    def start(self):
        os.sched_setaffinity(os.getpid(), self.cpu_list)
        isgpu = True if self.device != 'cpu' else False
        flag = get_message(self.conn, False)

        while flag != 'Stop':
            send_message(self.conn, 'break')

            if flag == 'Changed':
                info, _ = get_data(self.conn)
                s, e = info
                model = getEarlyExitChain(self.model, e)
                self.edge_model, _ = getPartitionedModels(model, s)

            model = self.edge_model

            x, down_latency = get_data(self.conn)

            get_message(self.conn, False)

            if x.device != self.device:
                x = x.to(self.device)
            ir, edge_latency = Inference(model, x, isgpu)

            edge_latency += down_latency

            send_data(self.conn, (ir, edge_latency))

            send_message(self.conn, 'break')

            flag = get_message(self.conn, False)
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