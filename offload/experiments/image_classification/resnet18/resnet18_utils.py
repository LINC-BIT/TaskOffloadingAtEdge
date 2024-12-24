import torch
import torch.nn as nn
import os
from offload.net.server import MultiDeployServer
from offload.net.client import MultiDeployClient
from offload.net.utils import send_short_data, get_message, get_short_data, send_message
from legodnn.utils.dl.common.model import set_module
import time
from tqdm import tqdm
from cv_task.image_classification.cifar.models.resnet import BasicBlock
from copy import deepcopy
import json

# 目前仅考虑去除模型的子一代中含有layer的Sequential
def remove_sequential(model):
    new_model = nn.Sequential()
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential) and len(module) != 0:
            # 展开 nn.Sequential 层为其包含的子层
            for s_name, submodule in module.named_children():
                new_name = f'{name}_{s_name}'
                new_model.add_module(new_name, submodule)
        else:
            new_model.add_module(name, module)  # 递归处理子模块
    return new_model

def getModelRootAndEnd(compressed_blocks_dir_path, device):
    frame = torch.load(os.path.join(compressed_blocks_dir_path, 'model_frame.pt'))
    root = nn.Sequential()
    end = nn.Sequential()
    rest = nn.Sequential()
    root_head = True
    frame = remove_sequential(frame)
    last_block = None
    last_block_name = ''
    for name, layer in frame.named_children():
        size = len([x for x in layer.parameters()])
        if (isinstance(layer, nn.Sequential) and size == 0) or isinstance(layer, BasicBlock): #BasicBlock必须为空
            if isinstance(layer, BasicBlock) and size > 0:
                raise ValueError('BasicBlock必须为空，请重新训练模型')
            if isinstance(layer, BasicBlock):
                last_block = layer
                last_block_name = name
            if root_head:
                root_head = False
            rest.add_module(name, layer)
        else:
            if root_head:
                root.add_module(name, layer)
            else:
                if last_block is not None:
                    for sname, l in last_block.named_children():
                        size = len([x for x in l.parameters()])
                        if (not isinstance(l, nn.Sequential)) and size == 0:
                            new_name = last_block_name.replace('.', '_') + '_' + sname
                            end.add_module(new_name, l)
                            set_module(last_block, sname, nn.Sequential())
                    last_block = None
                if isinstance(layer, nn.Linear):
                    end.add_module('flatten', nn.Flatten())
                end.add_module(name, layer)


    return root.to(device), end.to(device), rest.to(device)

def setOriginalBlock(frame, block):
    model = deepcopy(frame)
    for name, module in block.named_modules():
        if len(list(module.children())) > 0:
            continue
        if '.' in name:
            li = name.split('.')
            t = f'{li[0]}_{li[1]}'
            name = [t] + li[2:]
            name = '.'.join(name)
        block_module = deepcopy(module)
        set_module(model, name, block_module)
    new_block = nn.Sequential()
    for name, layer in model.named_children():
        size = len([x for x in layer.parameters()])
        if size == 0 and (isinstance(layer, nn.Sequential) or isinstance(layer, BasicBlock)):
            for sname, l in layer.named_children():
                size = len([x for x in l.parameters()])
                if (not isinstance(l, nn.Sequential)) and size == 0:
                    new_name = name.replace('.', '_') + '_' + sname
                    new_block.add_module(new_name, l)
        else:
            new_block.add_module(name, layer)
    new_block.eval()
    return new_block

class ResNetDeployServer(MultiDeployServer):
    def __init__(self, datasets, model_frame, deploy_info, block_manager,
                 trained_path, offload_path, device, sta_opt, running_flag, dict, bwdown, bwup):
        super().__init__(datasets, model_frame, deploy_info, block_manager, trained_path, device, dict, bwdown, bwup)
        self._offload_path = offload_path
        self._sta_opt = sta_opt
        self._running_flag = running_flag

    def _getblocks(self, block_id, sps):
        self._blocks_name = self._block_manager.get_blocks_id()
        self._block_dict = {}
        for id, sp in zip(block_id, sps):
            block = self._block_manager.get_block_from_file(
                os.path.join(self._trained_path, self._block_manager.get_block_file_name(self._blocks_name[id], sp)),
                self._device).to(self._device)
            block = setOriginalBlock(self._frame, block)
            self._block_dict[self._blocks_name[id]] = block

    def start(self):
        cnt = 0
        tested_num = 0
        correct = 0
        sum_infer_time = 0.
        acc_array = []
        l_array = []
        mem_array = []
        flops_array = []

        pbar = tqdm(total=self._len_dataset)
        for i, conn in enumerate(self.conns):
            send_short_data(conn, self._len_dataset, '推理任务数量', False)
            get_message(conn, False)

        changed = True
        while cnt < self._len_dataset:
            if self._sta_opt.value == 1:
                self._running_flag.value = 0
                pbar.set_description(f'Processing Task-{cnt + 1}')
                self._x, _y = next(self._datasets)
                if changed:
                    sign = 'Changed'
                    changed = False
                else:
                    sign = 'Start'
                self._startServer(sign)
                cnt += 1
                # logger.info('成功完成任务')
                # if cnt % 50 == 0:
                #     print(self._res)
                #     for i, bw in enumerate(self._bwdown):
                #         print(f"bandwidth monitor-{i} : get up bandwidth value : {self._bwup[i]:.3f} MB/s")
                #         print(f"bandwidth monitor-{i} : get down bandwidth value : {bw:.3f} MB/s")
                tested_num += self._x.shape[0]
                pbar.update(1)
                pred = torch.argmax(self._x, dim=1)
                correct += torch.sum(pred == _y).item()
                sum_infer_time += self._res['total_latency']
                acc_array.append(correct / tested_num)
                l_array.append(sum_infer_time / cnt)
                mem_array.append(self._res['model_size'] / (1024 ** 2))
                flops_array.append(self._res['FLOPs'] / 1e6)
                pbar.set_postfix(acc=acc_array[-1], avg_infer_time=sum_infer_time / cnt)
                time.sleep(0.005)
                self._running_flag.value = 1
            else:
                changed = True
        # drawPics(list(range(self._len_dataset)), acc_array, 'Avg Accuracy Curve', 'Task ID', 'Avg Accuracy', self._offload_path)
        # drawPics(list(range(self._len_dataset)), l_array, 'Avg Latency Curve', 'Task ID', 'Avg Latency (ms)', self._offload_path)
        # drawPics(list(range(self._len_dataset)), mem_array, 'Memory Curve', 'Task ID', 'Memory (MB)', self._offload_path)
        # drawPics(list(range(self._len_dataset)), flops_array, 'FLOPs Curve', 'Task ID', 'mFLOPs', self._offload_path)
        json.dump([acc_array, l_array, mem_array, flops_array], open(os.path.join(self._offload_path, 'legodnn.json'), 'w'))

class ResNetDeployClient(MultiDeployClient):
    def __init__(self, block_manager, rest_frame, trained_path, device, dict):
        super().__init__(block_manager, trained_path, device, dict)
        self._frame = rest_frame

    def _getblocks(self, block_id, sps):
        self._blocks_name = self._block_manager.get_blocks_id()
        self._block_dict = {}
        for id, sp in zip(block_id, sps):
            block = self._block_manager.get_block_from_file(
                os.path.join(self._trained_path, self._block_manager.get_block_file_name(self._blocks_name[id], sp)),
                self._device).to(self._device)
            block = setOriginalBlock(self._frame, block)
            self._block_dict[self._blocks_name[id]] = block

    def start(self):
        # os.sched_setaffinity(os.getpid(), list(range(self._used_core)))
        cnt = 0
        length = get_short_data(self.conn)
        pbar = tqdm(total=length)
        send_message(self.conn, 'break')

        while cnt < length:
            pbar.set_description(f'Processing Task-{cnt + 1}')
            self.startClient()
            cnt += 1
            pbar.update(1)

