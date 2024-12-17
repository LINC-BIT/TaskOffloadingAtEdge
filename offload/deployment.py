from scipy.optimize import minimize
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pprint
import os
import copy
import json
import pulp

from legodnn.abstract_block_manager import AbstractBlockManager
from legodnn.abstract_model_manager import AbstractModelManager
from legodnn.pure_runtime import PureRuntime
from offload.net.utils import get_speed
from offload.inference_utils import Inference
from offload.model import *
from offload.predictor.predict_tools import getLatency
from legodnn.utils.common.log import logger
from legodnn.utils.common.data_record import write_json
from legodnn.utils.common.file import ensure_dir, create_dir
from legodnn.utils.dl.common.model import save_model, ModelSaveMethod
from legodnn.utils.common.data_record import read_yaml

def divideModel(model, point):
    edge_model = nn.Sequential()
    cloud_model = nn.Sequential()
    for i, layer in enumerate(model):
        if i < point:
            edge_model.add_module(f'{i}-{layer.__class__.__name__}', layer)
        else:
            cloud_model.add_module(f'{i}-{layer.__class__.__name__}', layer)
    return edge_model, cloud_model


def getPartitionPoint(exist_latency:dict, bw:float):
    c_l = exist_latency['cloud']
    e_l = exist_latency['edge']
    ir_sizes = exist_latency['ir_sizes']
    point_num = len(c_l)
    best_latency = 0x3fffffff
    best_point = 0
    for i in range(point_num):
        trams_datasize = ir_sizes[i]
        sp = get_speed(bw)
        latency = e_l[i] + (trams_datasize / sp * 1000) + c_l[i]
        if latency < best_latency:
            best_latency = latency
            best_point = i
    return best_point

def getProfiles(rootpath:str, model, inputshape:tuple, is_edge_gpu:bool, is_cloud_gpu:bool, bw:float):
    dict = {'cloud':[], 'edge':[], 'ir_sizes':[]}
    for point in range(len(model) + 1):
        x = torch.rand(inputshape).to('cuda' if is_edge_gpu else 'cpu')
        edge_model, cloud_model = divideModel(model, point)
        edge_model = edge_model.to('cuda' if is_edge_gpu else 'cpu')
        cloud_model = cloud_model.to('cuda' if is_edge_gpu else 'cpu')
        # latency = getLatency(rootpath, edge_model, x, 'edge', is_edge_gpu) * x.shape[0]
        _, e_latency = Inference(edge_model, x, is_edge_gpu, 50)
        dict['edge'].append(e_latency)
        x = edge_model(x)
        c_latency = getLatency(rootpath, cloud_model, x, 'cloud', is_cloud_gpu)
        dict['cloud'].append(c_latency)
        if len(cloud_model) != 0:
            trans_datasize = torch.numel(x) * 4
        else:
            trans_datasize = 0
        dict['ir_sizes'].append(trans_datasize)

    return dict

class BlockProfiler:
    def __init__(self, block_manager: AbstractBlockManager, model_manager: AbstractModelManager,
                 trained_blocks_dir, offload_dir, test_sample_num, dummy_input_size,
                 device, platform):

        self._block_manager = block_manager
        self._model_manager = model_manager
        # self._composed_model = PureRuntime(trained_blocks_dir, block_manager, device)
        self._trained_blocks_dir = trained_blocks_dir
        self._offload_dir = offload_dir
        self._test_sample_num = test_sample_num
        self._dummy_input_size = dummy_input_size
        self._blocks_metrics_csv_path = os.path.join(offload_dir, f'{platform}-blocks-metrics.csv')
        self._teacher_model_metrics_yaml_path = os.path.join(offload_dir, f'{platform}-teacher-model-metrics.yaml')
        self._device = device

    def profile_original_blocks(self):
        if not os.path.exists(self._teacher_model_metrics_yaml_path):
            ensure_dir(self._teacher_model_metrics_yaml_path)
            blocks_id = self._block_manager.get_blocks_id()

            server_teacher_model_metrics = read_yaml(os.path.join(self._trained_blocks_dir,
                                                                  'server-teacher-model-metrics.yaml'))

            blocks_info = []
            for i, block_id in enumerate(blocks_id):
                # block_input_size = [1] + server_teacher_model_metrics['blocks_info'][i]['input_size']
                block_input_size = server_teacher_model_metrics['blocks_info'][i]['input_size']  # list or tuple
                if not isinstance(block_input_size, tuple):  # 单输入
                    input_data = torch.rand([1] + block_input_size).to(self._device)
                else:  # 多输入
                    input_data = ()
                    for tensor_size in block_input_size:
                        input_data = input_data + (torch.rand([1] + tensor_size).to(self._device),)
                input_data = (input_data,)

                raw_block = self._block_manager.get_block_from_file(
                    os.path.join(self._trained_blocks_dir,
                                 self._block_manager.get_block_file_name(block_id, 0.0)),
                    self._device
                )
                raw_block_latency = float(self._block_manager.get_block_latency(raw_block, self._test_sample_num, input_data,
                                                                          self._device))

                block_info = {
                    'index': i,
                    'id': block_id,
                    'latency': raw_block_latency
                }

                blocks_info += [block_info]
                logger.info('raw block info: {}'.format(json.dumps(block_info)))

            pure_runtime = PureRuntime(self._trained_blocks_dir, self._block_manager, self._device)
            pure_runtime.load_blocks([0.0 for _ in range(len(blocks_id))])
            teacher_model = pure_runtime.get_model()
            latency = float(self._model_manager.get_model_latency(teacher_model, self._test_sample_num,
                                                            self._dummy_input_size, self._device))

            obj = {
                'latency': latency,
                'blocks_info': blocks_info
            }

            self._teacher_model_metrics = obj

            with open(self._teacher_model_metrics_yaml_path, 'w') as f:
                yaml.dump(obj, f)
        else:
            logger.info('无需生成块的原始延迟信息')
    def profile_all_compressed_blocks(self):
        if not os.path.exists(self._blocks_metrics_csv_path):
            blocks_id = self._block_manager.get_blocks_id()
            blocks_num = len(blocks_id)

            server_teacher_model_metrics = read_yaml(os.path.join(self._trained_blocks_dir,
                                                                  'server-teacher-model-metrics.yaml'))

            latency_rel_drops = []

            for block_index in range(blocks_num):
                block_id = self._block_manager.get_blocks_id()[block_index]
                cur_raw_block = cur_block = self._block_manager.get_block_from_file(
                    os.path.join(self._trained_blocks_dir,
                                 self._block_manager.get_block_file_name(block_id, 0.0)),
                    self._device
                )
                # cur_block_input_size = [1] + server_teacher_model_metrics['blocks_info'][block_index]['input_size']
                cur_block_input_size = server_teacher_model_metrics['blocks_info'][block_index][
                    'input_size']  # list or tuple
                if not isinstance(cur_block_input_size, tuple):  # 单输入
                    input_data = torch.rand([1] + cur_block_input_size).to(self._device)
                else:  # 多输入
                    input_data = ()
                    for tensor_size in cur_block_input_size:
                        input_data = input_data + (torch.rand([1] + tensor_size).to(self._device),)
                input_data = (input_data,)

                # cur_raw_block_latency = self._block_manager.get_block_latency(cur_raw_block, self._test_sample_num, cur_block_input_size, self._device)
                cur_raw_block_latency = self._block_manager.get_block_latency(cur_raw_block, self._test_sample_num,
                                                                              input_data, self._device)

                for sparsity in self._block_manager.get_blocks_sparsity()[block_index]:
                    cur_block = self._block_manager.get_block_from_file(
                        os.path.join(self._trained_blocks_dir,
                                     self._block_manager.get_block_file_name(block_id, sparsity)),
                        self._device
                    )
                    # cur_block_latency = self._block_manager.get_block_latency(cur_block, self._test_sample_num, cur_block_input_size, self._device)
                    cur_block_latency = self._block_manager.get_block_latency(cur_block, self._test_sample_num, input_data,
                                                                              self._device)

                    if sparsity == 0.0:
                        cur_block_latency = cur_raw_block_latency
                    latency_rel_drop = (cur_raw_block_latency - cur_block_latency) / cur_raw_block_latency

                    logger.info('block {} (sparsity {}) latency rel drop: {:.3f}% '
                                '({:.3f}s -> {:.3f}s)'.format(block_id, sparsity, latency_rel_drop * 100,
                                                              cur_raw_block_latency, cur_block_latency))

                    latency_rel_drops += [latency_rel_drop]

            ensure_dir(self._blocks_metrics_csv_path)
            csv_file = open(self._blocks_metrics_csv_path, 'w')
            csv_file.write('block_index,block_sparsity,'
                           'test_accuracy_drop,inference_time_rel_drop,model_size_drop,FLOPs_drop,param_drop\n')
            i = 0
            for block_index in range(blocks_num):
                for sparsity in self._block_manager.get_blocks_sparsity()[block_index]:
                    csv_file.write('{},{},{},{},{},{},{}\n'.format(block_index, sparsity,
                                                                   0, latency_rel_drops[i], 0, 0, 0))
                    i += 1

            csv_file.close()
        else:
            logger.info('无需生成块压缩后的延迟信息')

    def profile_all_blocks(self):
        self.profile_original_blocks()
        self.profile_all_compressed_blocks()

class OptimalRuntime:
    def __init__(self,
                 trained_blocks_dir, offload_dir, model_input_size,
                 block_manager: AbstractBlockManager, model_manager: AbstractModelManager,
                 bwup, bwdown, latency_th, memory_th, device):

        logger.info('init adaptive model runtime')
        self._block_manager = block_manager
        self._model_manager = model_manager

        self._trained_blocks_dir = trained_blocks_dir
        self._model_input_size = model_input_size
        self._bwup = bwup
        self._bwdown = bwdown
        self._latency_th = latency_th
        self._memory_th = memory_th

        self._load_block_metrics(os.path.join(trained_blocks_dir, 'server-blocks-metrics.csv'),
                                 os.path.join(offload_dir, 'edge-blocks-metrics.csv'),
                                 os.path.join(offload_dir, 'cloud-blocks-metrics.csv'))
        self._load_model_metrics(os.path.join(trained_blocks_dir, 'server-teacher-model-metrics.yaml'),
                                 os.path.join(offload_dir, 'edge-teacher-model-metrics.yaml'),
                                 os.path.join(offload_dir, 'cloud-teacher-model-metrics.yaml'))

        self._xor_variable_id = 0
        self._last_selection = None
        self._selections_info = []
        self._before_first_adaption = True
        self._cur_original_model_infer_time = None

        self._progress_latancies()
        self._pure_runtime = PureRuntime(trained_blocks_dir, block_manager, device)
        self._init_model()

        self._cur_blocks_infer_time = None

    def _load_block_metrics(self, server_block_metrics_csv_path, edge_block_metrics_csv_file_path, cloud_block_metrics_csv_file_path):
        server_block_metrics_data = pd.read_csv(server_block_metrics_csv_path)
        server_block_metrics_data = np.asarray(server_block_metrics_data)
        edge_block_metrics_data = pd.read_csv(edge_block_metrics_csv_file_path)
        edge_block_metrics_data = np.asarray(edge_block_metrics_data)
        cloud_block_metrics_data = pd.read_csv(cloud_block_metrics_csv_file_path)
        cloud_block_metrics_data = np.asarray(cloud_block_metrics_data)

        self._acc_drops, self._edge_infer_time_rel_drops, self._cloud_infer_time_rel_drops, self._mem_drops, self._flops_drops, \
            self._param_drops = [], [], [], [], [], []

        total_block_index = 0
        for block_index in range(len(self._block_manager.get_blocks_sparsity())):
            this_block_num = len(self._block_manager.get_blocks_sparsity()[block_index])

            self._acc_drops.append(list(server_block_metrics_data[total_block_index: total_block_index + this_block_num, 2]))
            self._edge_infer_time_rel_drops.append(list(
                edge_block_metrics_data[total_block_index: total_block_index + this_block_num, 3]))
            self._cloud_infer_time_rel_drops.append(list(
                cloud_block_metrics_data[total_block_index: total_block_index + this_block_num, 3]))
            self._mem_drops.append(list(server_block_metrics_data[total_block_index: total_block_index + this_block_num, 4]))
            self._flops_drops.append((
                server_block_metrics_data[total_block_index: total_block_index + this_block_num, 5]))
            self._param_drops.append((
                server_block_metrics_data[total_block_index: total_block_index + this_block_num, 6]))

            total_block_index += this_block_num

        self._acc_drops, self._mem_drops, self._flops_drops, self._param_drops, self._edge_infer_time_rel_drops, self._cloud_infer_time_rel_drops = \
            np.asarray(self._acc_drops), np.asarray(self._mem_drops), np.asarray(self._flops_drops), np.asarray(self._param_drops), np.asarray(self._edge_infer_time_rel_drops), np.asarray(self._cloud_infer_time_rel_drops)

        logger.info('load blocks metrics')

    def _load_model_metrics(self, server_model_metrics_yaml_file_path, edge_model_metrics_yaml_file_path, cloud_model_metrics_yaml_file_path):
        server_f = open(server_model_metrics_yaml_file_path, 'r')
        server_original_model_metrics = yaml.load(server_f, yaml.Loader)
        edge_f = open(edge_model_metrics_yaml_file_path, 'r')
        edge_original_model_metrics = yaml.load(edge_f, yaml.Loader)
        cloud_f = open(cloud_model_metrics_yaml_file_path, 'r')
        cloud_original_model_metrics = yaml.load(cloud_f, yaml.Loader)

        self._original_model_acc = server_original_model_metrics['test_accuracy']
        self._original_model_size = server_original_model_metrics['model_size']

        self._original_model_flops = server_original_model_metrics['FLOPs']
        self._original_model_param = server_original_model_metrics['param']
        self._model_input_size = np.cumprod(server_original_model_metrics['blocks_info'][0]['input_size'])[-1] * 4 / (1024 ** 2)
        self._original_blocks_size = np.array(
            list(map(lambda i: i['size'], server_original_model_metrics['blocks_info'])))

        self._model_output_sizes = np.array(
            list(map(lambda i: np.cumprod(i['output_size'])[-1] * 4 / 1024 / 1024, server_original_model_metrics['blocks_info'])))

        self._edge_original_blocks_infer_time = np.array(
            list(map(lambda i: i['latency'], edge_original_model_metrics['blocks_info'])))
        self._edge_original_model_infer_time = edge_original_model_metrics['latency']

        self._cloud_original_blocks_infer_time = np.array(
            list(map(lambda i: i['latency'], cloud_original_model_metrics['blocks_info'])))
        self._cloud_original_model_infer_time = cloud_original_model_metrics['latency']

        self._blocks_input_size = list(map(lambda i: i['input_size'], server_original_model_metrics['blocks_info']))

        blocks_sparsity = self._block_manager.get_blocks_sparsity()
        original_blocks_size = []
        for i in range(len(blocks_sparsity)):
            original_blocks_size .append([self._original_blocks_size[i] for _ in range(len(blocks_sparsity[i]))])
        original_blocks_size = np.asarray(original_blocks_size)
        self._blocks_size = original_blocks_size - self._mem_drops

        server_f.close()
        edge_f.close()

        logger.info('load model metrics')

    def _progress_latancies(self):
        ratio = 1 - self._cloud_infer_time_rel_drops
        matrix = np.tile(self._cloud_original_blocks_infer_time[:, None], (1, ratio.shape[1]))
        self._cloud_block_infer_times = matrix * ratio

        ratio = 1 - self._edge_infer_time_rel_drops
        matrix = np.tile(self._edge_original_blocks_infer_time[:, None], (1, ratio.shape[1]))
        self._edge_block_infer_times = matrix * ratio

    def _init_model(self):
        sparsest_selection = np.zeros_like(self._mem_drops)
        devices = [0 for _ in range(self._mem_drops.shape[0])]

        blocks_sparsity = self._block_manager.get_blocks_sparsity()
        for block_index in range(len(blocks_sparsity)):
            sparsest_selection[block_index, -1] = 1

        logger.info('load sparest blocks for initializing model')
        self._apply_selection_and_devices_to_model(sparsest_selection, devices)
        self._least_model_size = self._model_manager.get_model_size(self._pure_runtime.get_model())
        self._last_selection = sparsest_selection

    def _get_readable_block_selection(self, selection):
        chosen_blocks_sparsity = []
        flatten_blocks_sparsity = []
        for s in self._block_manager.get_blocks_sparsity():
            flatten_blocks_sparsity.append(s)

        for item, s in zip(selection, flatten_blocks_sparsity):
            id = np.where(item == 1)[0][0]
            chosen_blocks_sparsity += [s[id]]

        return chosen_blocks_sparsity

    def _apply_selection_and_devices_to_model(self, cur_selection, devices):
        def get_adaption_swap_mem_cost():
            if self._before_first_adaption:
                return 0
            return np.sum(np.logical_xor(self._last_selection, cur_selection) * self._blocks_size)

        transmit_latency = 0.
        for id in range(len(devices)):
            if devices[id] == 1:
                if id == 0:
                    transmit_latency += self._model_input_size / self._bwup.value

                if id == len(devices) - 1:
                    transmit_latency += self._model_output_sizes[id] / self._bwdown.value

            if devices[id] != devices[id - 1]:
                transmit_latency += devices[id - 1] * (1 - devices[id]) * self._model_output_sizes[id - 1] / self._bwdown.value + \
                                    (1 - devices[id - 1]) * devices[id] * self._model_output_sizes[id - 1] / self._bwup.value

        adaption_swap_mem_cost = get_adaption_swap_mem_cost()
        adaption_time_cost = transmit_latency
        return adaption_swap_mem_cost, adaption_time_cost

    def _sp2matrix(self, sp):
        sp_list = self._block_manager.get_blocks_sparsity()[0]
        tmp = np.eye(len(sp_list))
        mat = np.zeros((len(sp), len(sp_list)))
        for row, i in enumerate(sp):
            flag = False
            for row2, s in enumerate(sp_list):
                if s > i:
                    flag = True
                    mat[row] = tmp[row2 - 1]
                    break
            if not flag:
                mat[row] = tmp[row2]

        return mat

    def update_model(self, cur_max_infer_time, cur_max_model_size, max_edge_infer_time, max_edge_model_size):
        cur_max_model_size *= 1024 ** 2
        max_edge_model_size *= 1024 ** 2

        logger.info('cur max inference time: {:.6f}s, '
                    'cur available max memory: {}B ({:.3f}MB), '
                    'try to adapt blocks'.format(cur_max_infer_time, cur_max_model_size,
                                                 cur_max_model_size / 1024 ** 2))

        def sigmoid(x):
            x = (x - 0.5) * 4
            return 1 / (1 + np.exp(-x))

        def rqs(args):
            a, m, o_size, maxsize, el, cl, bwup, bwdown, sizes, maxtime, input_size = args

            def obj(v):
                v = np.reshape(v, (a.shape[0], -1))
                d, s = v[:, 0], self._sp2matrix(v[:, 1])
                tar1 = np.sum(s * a)
                tar2 = con2(v)
                tar3 = con3(v)
                return tar1 + tar2 + tar3

            def con1(v): #eq
                v = np.reshape(v, (m.shape[0], -1))
                d, s = v[:, 0], v[:, 1]
                tar = np.sum(d - np.floor(d))
                return tar

            def con2(v): #ineq
                v = np.reshape(v, (m.shape[0], -1))
                d, s = v[:, 0], self._sp2matrix(v[:, 1])
                tar = maxsize + (np.sum(s * m) - o_size)
                return tar

            def con3(v): #ineq
                v = np.reshape(v, (m.shape[0], -1))
                d, s = np.round(v[:, 0]), self._sp2matrix(v[:, 1])
                td = np.tile(d[:, None], (1, s.shape[1]))
                tmp1 = np.sum(s * ((1 - td) * el + td * cl))
                tmp2 = np.sum(d[:-1] * (1 - d[1:]) * (sizes[:-1] / bwdown) + (1 - d[:-1]) * d[1:] * (sizes[:-1] / bwup))
                tmp3 = d[-1] * (sizes[-1] / bwdown) + d[0] * (input_size / bwup)
                tar = maxtime - tmp1 - tmp2 - tmp3
                return tar

            def con4(v): #ineq
                v = np.reshape(v, (m.shape[0], -1))
                d, s = np.round(v[:, 0]), self._sp2matrix(v[:, 1])
                td = np.tile(d[:, None], (1, s.shape[1]))
                tar = max_edge_infer_time - np.sum((1 - td) * s * el)
                return tar

            def con5(v): #ineq
                v = np.reshape(v, (m.shape[0], -1))
                d, s = np.round(v[:, 0]), self._sp2matrix(v[:, 1])
                td = np.tile(d[:, None], (1, s.shape[1]))
                tar = max_edge_model_size - np.sum((1 - td) * s * m)
                return tar

            cons_dicts = (
                {'type': 'eq', 'fun': con1},
                {'type': 'ineq', 'fun': con2},
                {'type': 'ineq', 'fun': con3},
                {'type': 'ineq', 'fun': con4},
                {'type': 'ineq', 'fun': con5},
            )

            return obj, cons_dicts

        block_num, sp_num = self._mem_drops.shape
        # varibles = np.concatenate([np.ones((block_num, 1)), self._last_selection], axis=1).reshape((-1))
        varibles = np.concatenate([sigmoid(np.random.random((block_num, 1))), np.random.random((block_num, 1))], axis=1).reshape(-1)
        bounds = [(0, 1) for _ in range(block_num * 2)]
        args = (self._acc_drops, self._mem_drops, self._original_model_size, cur_max_model_size, self._edge_block_infer_times,
                 self._cloud_block_infer_times, self._bwup.value, self._bwdown.value, self._model_output_sizes, cur_max_infer_time,
                self._model_input_size)
        fun, cons = rqs(args)
        res = minimize(fun, varibles, method='SLSQP', constraints=cons, options={}, bounds=bounds, tol=1e-8)

        varibles = np.reshape(res.x, (block_num, 2))
        judge = [0., 0., 0., 0., 0.]
        for i in range(len(judge)):
            if i == 0:
                judge[i] = np.abs(cons[i]['fun'](varibles))
            else:
                judge[i] = cons[i]['fun'](varibles)
        while judge[1] < 0 or judge[2] < 0 or judge[3] < 0 or judge[4] < 0:
            # print(judge)
            varibles = np.concatenate([sigmoid(np.random.random((block_num, 1))), np.random.random((block_num, 1))], axis=1).reshape(-1)
            res = minimize(fun, varibles, method='SLSQP', constraints=cons, options={}, bounds=bounds, tol=1e-8)
            varibles = np.reshape(res.x, (block_num, 2))
            for i in range(len(judge)):
                if i == 0:
                    judge[i] = np.abs(cons[i]['fun'](varibles))
                else:
                    judge[i] = cons[i]['fun'](varibles)

        d, s = np.round(varibles[:, 0]), self._sp2matrix(varibles[:, 1])
        devices, selection = self._get_readable_block_device(d), self._get_readable_block_selection(s)

        acc = self._original_model_acc - np.sum(s * self._acc_drops)
        flops = self._original_model_flops - np.sum(s * self._flops_drops)
        block_adaption_mem_swap, block_adaption_time = self._apply_selection_and_devices_to_model(s, d)
        self._last_selection = selection
        self._before_first_adaption = False

        selection_info = {
            'blocks_sparsity': selection,
            'blocks_devices': devices,
            'esti_test_accuracy': acc,
            'esti_latency': cur_max_infer_time - judge[2],
            'model_size': cur_max_model_size - judge[1],
            'FLOPs': flops,
            'update_swap_mem_cost': block_adaption_mem_swap,
            'update_transmit_time_cost': block_adaption_time,

        }

        self._selections_info += [selection_info]
        return selection_info

    def _get_readable_block_device(self, d):
        res = []
        for de in d:
            if de == 1:
                res.append('cloud')
            else:
                res.append('edge')
        return res

    def get_selections_info(self):
        return self._selections_info

class MultiOptimalRuntime:
    def __init__(self,
                 trained_blocks_dir, model_input_size,
                 block_manager: AbstractBlockManager, model_manager: AbstractModelManager, deploy_infos,
                 bwups, bwdowns, l_ths, m_ths, f_array, acc_thres, device, clients_name, pulp_solver=pulp.PULP_CBC_CMD(msg=False, gapAbs=0)):

        logger.info('init adaptive model runtime')
        self._block_manager = block_manager
        self._model_manager = model_manager

        self._trained_blocks_dir = trained_blocks_dir
        self._model_input_size = model_input_size
        self._model_input_shape = model_input_size
        self._bwups = bwups
        self._bwdowns = bwdowns
        self._l_ths = l_ths
        self._m_ths = m_ths
        self._f_array = f_array
        self._deploy_infos = deploy_infos
        self._pulp_solver = pulp_solver

        self._load_block_metrics(os.path.join(trained_blocks_dir, 'server-blocks-metrics.csv'))
        self._load_model_metrics(os.path.join(trained_blocks_dir, 'server-teacher-model-metrics.yaml'))

        self._xor_variable_id = 0
        self._device_num = len(clients_name) + 1
        self._last_selection = None
        self._selections_info = []
        self._before_first_adaption = True
        self._cur_original_model_infer_time = None
        self._latency_thres = None
        self._memory_thres = None
        self._acc_thres = acc_thres
        # self._progress_latancies()
        self._pure_runtime = PureRuntime(trained_blocks_dir, block_manager, device)
        self._init_model()

        self._cur_blocks_infer_time = None

    def _load_block_metrics(self, server_block_metrics_csv_path):
        server_block_metrics_data = pd.read_csv(server_block_metrics_csv_path)
        server_block_metrics_data = np.asarray(server_block_metrics_data)

        # edge_block_metrics_datas = []
        # for name in client_names:
        #     edge_block_metrics_data = pd.read_csv(edge_block_metrics_csv_file_path + '_' + name + '.csv')
        #     edge_block_metrics_datas.append(np.asarray(edge_block_metrics_data))
        #
        # cloud_block_metrics_data = pd.read_csv(cloud_block_metrics_csv_file_path)
        # cloud_block_metrics_data = np.asarray(cloud_block_metrics_data)

        self._acc_drops, self._edge_infer_time_rel_drops, self._cloud_infer_time_rel_drops, self._mem_drops, self._flops_drops, \
            self._param_drops = [], [], [], [], [], []

        # for id in range(len(client_names)):
        #     self._edge_infer_time_rel_drops.append(list())

        total_block_index = 0
        for block_index in range(len(self._block_manager.get_blocks_sparsity())):
            this_block_num = len(self._block_manager.get_blocks_sparsity()[block_index])

            self._acc_drops.append(list(server_block_metrics_data[total_block_index: total_block_index + this_block_num, 2]))

            # for id in range(len(client_names)):
            #     self._edge_infer_time_rel_drops[id].append(list(
            #         edge_block_metrics_datas[id][total_block_index: total_block_index + this_block_num, 3]))
            #
            # self._cloud_infer_time_rel_drops.append(list(
            #     cloud_block_metrics_data[total_block_index: total_block_index + this_block_num, 3]))
            self._mem_drops.append(list(server_block_metrics_data[total_block_index: total_block_index + this_block_num, 4]))
            self._flops_drops.append((
                server_block_metrics_data[total_block_index: total_block_index + this_block_num, 5]))
            self._param_drops.append((
                server_block_metrics_data[total_block_index: total_block_index + this_block_num, 6]))

            total_block_index += this_block_num

        self._acc_drops, self._mem_drops, self._flops_drops, self._param_drops, self._edge_infer_time_rel_drops, self._cloud_infer_time_rel_drops = \
            np.asarray(self._acc_drops), np.asarray(self._mem_drops), np.asarray(self._flops_drops), np.asarray(self._param_drops), np.asarray(self._edge_infer_time_rel_drops), np.asarray(self._cloud_infer_time_rel_drops)

        logger.info('load blocks metrics')

    def _load_model_metrics(self, server_model_metrics_yaml_file_path):
        server_f = open(server_model_metrics_yaml_file_path, 'r')
        server_original_model_metrics = yaml.load(server_f, yaml.Loader)

        # edge_original_model_metrics = []
        # edge_fs = []

        # for clinet_name in clinet_names:
        #     edge_f = open(edge_model_metrics_yaml_file_path + '_' + clinet_name + '.yaml', 'r')
        #     edge_original_model_metrics.append(yaml.load(edge_f, yaml.Loader))
        #     edge_fs.append(edge_f)
        #
        # cloud_f = open(cloud_model_metrics_yaml_file_path, 'r')
        # cloud_original_model_metrics = yaml.load(cloud_f, yaml.Loader)

        self._original_model_acc = server_original_model_metrics['test_accuracy']
        self._original_model_size = server_original_model_metrics['model_size']

        self._original_model_flops = server_original_model_metrics['FLOPs']
        self._original_model_param = server_original_model_metrics['param']

        self._original_blocks_flops = np.array(
            list(map(lambda i: i['FLOPs'], server_original_model_metrics['blocks_info'])))

        self._model_input_size = np.cumprod(server_original_model_metrics['blocks_info'][0]['input_size'])[-1] * 4 / (1024 ** 2)
        self._original_blocks_size = np.array(
            list(map(lambda i: i['size'], server_original_model_metrics['blocks_info'])))

        self._model_output_sizes = np.array(
            list(map(lambda i: np.cumprod(i['output_size'])[-1] * 4 / 1024 / 1024, server_original_model_metrics['blocks_info'])))


        # self._edge_original_blocks_infer_time = np.array(
        #     [list(map(lambda i: i['latency'], edge_original_model_metric['blocks_info'])) for edge_original_model_metric in edge_original_model_metrics])
        # self._edge_original_model_infer_time = np.array([edge_original_model_metric['latency'] for edge_original_model_metric in edge_original_model_metrics])
        #
        # self._cloud_original_blocks_infer_time = np.array(
        #     list(map(lambda i: i['latency'], cloud_original_model_metrics['blocks_info'])))
        # self._cloud_original_model_infer_time = cloud_original_model_metrics['latency']

        self._blocks_input_size = list(map(lambda i: i['input_size'], server_original_model_metrics['blocks_info']))

        blocks_sparsity = self._block_manager.get_blocks_sparsity()
        original_blocks_size = []
        for i in range(len(blocks_sparsity)):
            original_blocks_size.append([self._original_blocks_size[i] for _ in range(len(blocks_sparsity[i]))])
        original_blocks_size = np.asarray(original_blocks_size)
        self._blocks_size = original_blocks_size - self._mem_drops

        server_f.close()
        # for edge_f in edge_fs:
        #     edge_f.close()

        logger.info('load model metrics')

    # def _progress_latancies(self):
    #     ratio = 1 - self._cloud_infer_time_rel_drops
    #     matrix = np.tile(self._cloud_original_blocks_infer_time[:, None], (1, ratio.shape[1]))
    #     self._cloud_block_infer_times = matrix * ratio
    #
    #     ratio = 1 - self._edge_infer_time_rel_drops
    #     matrix = np.tile(self._edge_original_blocks_infer_time[:, :, None], (1, 1, ratio.shape[2]))
    #     self._edge_block_infer_times = matrix * ratio

    def _init_model(self):
        sparsest_selection = np.zeros_like(self._mem_drops).reshape(-1)

        i = 0
        blocks_sparsity = self._block_manager.get_blocks_sparsity()
        for block_index in range(len(blocks_sparsity)):
            sparsest_selection[i + len(blocks_sparsity[block_index]) - 1] = 1
            i += len(blocks_sparsity[block_index])

        logger.info('load sparest blocks for initializing model')
        self._least_model_size = self._model_manager.get_model_size(self._pure_runtime.get_model())
        self._last_selection = sparsest_selection

    def update_model(self):
        assert self._latency_thres != None and self._memory_thres != None, '请指定当前推理的约束'

        def apply(selection, drop):
            return pulp.lpDot(selection, drop)

        def solve(l_gamma=1., m_gamma=1.):
            pos_xy = lambda i, b: i * (block_num - 1) + b
            pos_w = lambda i, j, b: i * (self._device_num - 1) * (block_num - 1) + j * (block_num - 1) + b
            pos_p = lambda i, b, sp: i * block_num * sp_num + b * sp_num + sp
            pos_s = lambda b, sp: b * sp_num + sp
            pos_e = lambda i, b: i * block_num + b

            act_acc_drop = apply(s, self._acc_drops.reshape(-1))  # 精度下降
            constraints = []
            download_input_latency = pulp.lpSum([e[pos_e(i, 0)] * self._model_input_size /
                                                   self._bwdowns[i - 1] for i in range(1, self._device_num)])
            upload_output_latency = pulp.lpSum([e[pos_e(i, block_num - 1)] * self._model_output_sizes[-1] /
                                                   self._bwups[i - 1] for i in range(1, self._device_num)])
            transmit_latency = download_input_latency + apply(x, c2e_l) + apply(y, e2c_l) + apply(w, e2e_l) + upload_output_latency
            infer_latency = pulp.lpSum([apply(p[pos_p(i, 0, 0): pos_p(i, block_num - 1, sp_num - 1)], flops_mat.reshape(-1)) /
                            self._f_array[i] for i in range(self._device_num)])
            device_latency = [apply(s, flops_mat.reshape(-1)) * self._model_input_shape[0] / self._f_array[i] for i in range(self._device_num)]
            device_mem = [pulp.lpSum(apply(p[pos_p(id, 0, 0): pos_p(id, block_num - 1, sp_num - 1)], self._blocks_size.reshape(-1))) for id in range(self._device_num)]
            act_latency = (infer_latency + transmit_latency) * self._model_input_shape[0]
            act_mem_drops = apply(s, self._mem_drops.reshape(-1))
            act_flops_drop = apply(s, self._flops_drops.reshape(-1))

            # 目标函数
            obj = -act_mem_drops + act_latency + apply(s, self._acc_drops.reshape(-1))

            # 基本约束
            # constraints += [self._latency_thres >= act_latency]
            constraints += [self._original_model_acc - act_acc_drop >= self._acc_thres]
            constraints += [l_gamma * device_latency[0] >= act_latency]
            # constraints += [self._original_model_size - act_mem_drops <= self._memory_thres]
            constraints += [device_mem[0] <= self._memory_thres]

            # 设备约束
            for block in range(block_num):
                constraints += [pulp.lpSum([e[pos_e(index, block)] for index in range(self._device_num)]) == 1]

            # 块大小选择约束
            for index in range(0, block_num * sp_num, sp_num):
                constraints += [pulp.lpSum(s[index: index + sp_num]) == 1]

            # 非线性转线性约束
            # for x:
            for i in range(1, self._device_num):
                for b in range(block_num - 1):
                    constraints.append(x[pos_xy(i - 1, b)] <= e[pos_e(0, b)])
                    constraints.append(x[pos_xy(i - 1, b)] <= e[pos_e(i, b+1)])
                    constraints.append(x[pos_xy(i - 1, b)] >= e[pos_e(0, b)] + e[pos_e(i, b+1)] - 1)

            # for y:
            for i in range(1, self._device_num):
                for b in range(block_num - 1):
                    constraints.append(y[pos_xy(i - 1, b)] <= e[pos_e(i, b)])
                    constraints.append(y[pos_xy(i - 1, b)] <= e[pos_e(0, b+1)])
                    constraints.append(y[pos_xy(i - 1, b)] >= e[pos_e(i, b)] + e[pos_e(0, b+1)] - 1)

            # for w:
            for i in range(1, self._device_num):
                for j in range(1, self._device_num):
                    for b in range(block_num - 1):
                        constraints.append(w[pos_w(i - 1, j - 1, b)] <= e[pos_e(i, b)])
                        constraints.append(w[pos_w(i - 1, j - 1, b)] <= e[pos_e(j, b+1)])
                        constraints.append(w[pos_w(i - 1, j - 1, b)] >= e[pos_e(i, b)] + e[pos_e(j, b+1)] - 1)

            # for p:
            for i in range(self._device_num):
                for b in range(block_num):
                    for sp in range(sp_num):
                        constraints.append(p[pos_p(i, b, sp)] <= e[pos_e(i, b)])
                        constraints.append(p[pos_p(i, b, sp)] <= s[pos_s(b, sp)])
                        constraints.append(p[pos_p(i, b, sp)] >= e[pos_e(i, b)] + s[pos_s(b, sp)] - 1)

            selection, devices = _lp_solve(obj, constraints)

            return selection, devices, act_acc_drop, act_latency, act_mem_drops, act_flops_drop, transmit_latency, device_latency[0], device_mem[0]

        def _lp_solve(obj, constraints):
            prob = pulp.LpProblem('device-block-selection-optimization', pulp.LpMinimize)

            prob += obj
            for con in constraints:
                prob += con

            # logger.info('solving...')
            status = prob.solve(self._pulp_solver)
            # logger.info('solving finished')

            if status != 1:
                return None, None
            else:
                sel = prob.variables()
                sel = list(filter(lambda v: not v.name.startswith('xor') and not v.name.startswith('__dummy'), sel))

                device = [0. for i in range(block_num)]
                selection = []
                p = []

                for v in sel:
                    if 'e' in v.name:
                        de, bl = [int(_) for _ in v.name.split('_')[1:]]
                        if v.varValue.real == 1:
                            device[bl] = de
                    elif 's' in v.name:
                        selection.append(v.varValue.real)
                    elif 'p' in v.name and v.varValue.real > 0:
                        p.append(v.name)
                return selection, device

        l_gamma = 1.
        m_gamma = 1.

        is_relaxed = False

        block_num, sp_num = self._mem_drops.shape

        c2e_l = np.array([self._model_output_sizes[b] / self._bwdowns[i - 1]
                          for i in range(1, self._device_num) for b in range(block_num - 1)])
        e2c_l = np.array([self._model_output_sizes[b] / self._bwups[i - 1]
                          for i in range(1, self._device_num) for b in range(block_num - 1)])
        e2e_l = np.array([self._model_output_sizes[b] / self._bwdowns[i - 1] + self._model_output_sizes[b] / self._bwups[i - 1]
                          if i != j else 0 for i in range(1, self._device_num) for j in range(1, self._device_num)
                          for b in range(block_num - 1)])

        flops_mat = self._original_blocks_flops[:, None] - self._flops_drops

        e = [pulp.LpVariable(f'e_{i}_{b}', lowBound=0, upBound=1, cat=pulp.LpBinary)
             for i in range(self._device_num) for b in range(block_num)]
        s = [pulp.LpVariable(f's_{b}_{sp}', lowBound=0, upBound=1, cat=pulp.LpBinary)
             for b in range(block_num) for sp in range(sp_num)]
        x = [pulp.LpVariable(f'x_{i}_{b}', lowBound=0, upBound=1, cat=pulp.LpBinary)
             for i in range(1, self._device_num) for b in range(block_num - 1)]
        y = [pulp.LpVariable(f'y_{i}_{b}', lowBound=0, upBound=1, cat=pulp.LpBinary)
             for i in range(1, self._device_num) for b in range(block_num - 1)]
        w = [pulp.LpVariable(f'w_{i}_{j}_{b}', lowBound=0, upBound=1, cat=pulp.LpBinary)
             for i in range(1, self._device_num) for j in range(1, self._device_num) for b in range(block_num - 1)]
        p = [pulp.LpVariable(f'p_{i}_{b}_{sp}', lowBound=0, upBound=1, cat=pulp.LpBinary)
             for i in range(self._device_num) for b in range(block_num) for sp in range(sp_num)]

        selection, devices, act_acc_drop, act_latency, act_mem_drops, act_flops_drop, transmit_latency, server_device_latency, server_device_mem = solve(l_gamma, m_gamma)

        logger.info('cur predict inference time: {:.6f}s, '
                    'cur memory: {}B ({:.3f}MB), '
                    'try to adapt blocks'.format(pulp.value(act_latency), self._original_model_size - pulp.value(act_mem_drops),
                                                 (self._original_model_size - pulp.value(act_mem_drops)) / 1024 ** 2))
        # mem = pulp.value(server_device_mem)
        # while selection is None and pulp.value(server_device_mem) > self._memory_thres:
        #     is_relaxed = True
        #     self._memory_thres += 0.1 * 1024 ** 2
        #     if self._memory_thres < self._least_model_size:
        #         self._memory_thres = self._least_model_size
        #
        #     l_gamma += 0.1
        #
        #     logger.info('no solution found, relax the memory constraint '
        #                 'to {}B ({:.3f}MB) and continue finding solution'.format(self._memory_thres,
        #                                                                          self._memory_thres / 1024 ** 2))
        #     selection, devices, act_acc_drop, act_latency, act_mem_drops, act_flops_drop, transmit_latency, server_device_latency, server_device_mem = solve(l_gamma, m_gamma)

        # latency = pulp.value(act_latency)
        # while selection is None and pulp.value(act_latency) > self._latency_thres:
        #     is_relaxed = True
        #     self._latency_thres += 0.1
        #
        #     logger.info('no solution found, relax the time constraint to {:.6f}s and '
        #                 'continue finding solution'.format(self._latency_thres))
        #     selection, devices, act_acc_drop, act_latency, act_mem_drops, act_flops_drop, transmit_latency, server_device_latency, server_device_mem = solve(l_gamma, m_gamma)
        s_d_l = l_gamma * pulp.value(server_device_latency)
        while selection is None and l_gamma * pulp.value(server_device_latency) < pulp.value(act_latency):
            is_relaxed = True
            l_gamma += 0.1

            logger.info('no solution found, relax the server device latency constraint and '
                        'continue finding solution')
            selection, devices, act_acc_drop, act_latency, act_mem_drops, act_flops_drop, transmit_latency, server_device_latency, server_device_mem = solve(l_gamma, m_gamma)

        s_d_m = pulp.value(server_device_mem)
        while selection is None and pulp.value(server_device_mem) < self._memory_thres:
            is_relaxed = True
            self._memory_thres += 0.1 * 1024 ** 2

            logger.info('no solution found, relax the memory constraint '
                        'to {}B ({:.3f}MB) and continue finding solution'.format(self._memory_thres,
                                                                                 self._memory_thres / 1024 ** 2))
            selection, devices, act_acc_drop, act_latency, act_mem_drops, act_flops_drop, transmit_latency, server_device_latency, server_device_mem = solve(
                l_gamma, m_gamma)

        acc = self._original_model_acc - pulp.value(act_acc_drop)
        flops = self._original_model_flops - pulp.value(act_flops_drop)
        latency = pulp.value(act_latency)
        mem = self._original_model_size - pulp.value(act_mem_drops)
        transmit_time_cost = pulp.value(transmit_latency)
        readable_selection = self._get_readable_block_selection(selection)
        readable_devices = self._get_readable_block_device(devices)
        self._last_selection = selection
        self._before_first_adaption = False

        selection_info = {
            'blocks_sparsity': selection,
            'readable_sparsity': readable_selection,
            'blocks_devices': devices,
            'readable_blocks_devices': readable_devices,
            'esti_test_accuracy': acc,
            'esti_latency': latency,
            'model_size': mem,
            'FLOPs': flops,
            # 'update_swap_mem_cost': block_adaption_mem_swap,
            'all_transmit_time_cost': transmit_time_cost,
            'relaxed': is_relaxed
        }

        self._deploy_infos['infos'] = selection_info

    def _get_readable_block_selection(self, selection):
        chosen_blocks_sparsity = []
        flatten_blocks_sparsity = []
        for s in self._block_manager.get_blocks_sparsity():
            flatten_blocks_sparsity += s

        for item, s in zip(selection, flatten_blocks_sparsity):
            if item > 0:
                chosen_blocks_sparsity += [s]

        return chosen_blocks_sparsity

    def _get_readable_block_device(self, d):
        res = []
        for de in d:
            if de == 0:
                res.append('cloud')
            else:
                res.append(f'edge-{de - 1}')
        return res

    def setCurInferThresholds(self, l_th, m_th):
        self._latency_thres = l_th
        self._memory_thres = m_th

def gen_series_legodnn_models(deadline, model_size_search_range, target_model_num, optimal_runtime,
                              descendant_models_save_path, save_model_flag=False):
    min_model_size = model_size_search_range[0] * 1024 ** 2
    max_model_size = model_size_search_range[1] * 1024 ** 2

    logger.info('min model size: {:.3f}MB, max model size: {:.3f}MB'.format(min_model_size / 1024 ** 2,
                                                                            max_model_size / 1024 ** 2))
    # target_model_num = 50

    models_info = []
    create_dir(descendant_models_save_path)

    models_info_path = os.path.join(descendant_models_save_path, 'models-info.json')
    blocks_sparsity_list = []  # 去掉重复的blocks_sparsity
    num = 0
    for i, target_model_size in enumerate(np.linspace(min_model_size, max_model_size, target_model_num)):

        logger.info('target model size: {:.3f}MB'.format(target_model_size / 1024 ** 2))

        update_info = optimal_runtime.update_model(deadline, target_model_size)
        if update_info['blocks_sparsity'] not in blocks_sparsity_list:
            blocks_sparsity_list.append(update_info['blocks_sparsity'])
            logger.info('update info: \n{}'.format(pprint.pformat(update_info, indent=2, depth=2)))
            cur_model = copy.deepcopy(optimal_runtime._pure_runtime.get_model())
            cur_model = cur_model.to('cpu')
            # if hasattr(cur_model, 'forward_dummy'):
            #     cur_model.forward = cur_model.forward_dummy

            # model_save_path = os.path.join(descendant_models_save_path, '{}.jit'.format(num))
            model_save_path = os.path.join(descendant_models_save_path, '{}.pt'.format(num))
            # save_model(cur_model, model_save_path, ModelSaveMethod.JIT, optimal_runtime._model_input_size)
            if save_model_flag:
                save_model(cur_model, model_save_path, ModelSaveMethod.FULL, optimal_runtime._model_input_size)
            models_info += [{
                # 'model_file_name': '{}.jit'.format(num),
                'model_file_name': '{}.pt'.format(num),
                'model_info': update_info
            }]

            write_json(models_info_path, models_info, backup=False)
            num = num + 1
    # visualize
    models_size = [i['model_info']['model_size'] / 1024 ** 2 for i in models_info]
    models_acc = [i['model_info']['esti_test_accuracy'] for i in models_info]

    plt.rc('font', family='Times New Roman')
    plt.rcParams['font.size'] = '20'
    plt.plot(models_size, models_acc, linewidth=2, color='black')
    plt.xlabel('model size (MB)')
    plt.ylabel('acc')
    plt.tight_layout()
    plt.savefig(os.path.join(descendant_models_save_path, 'models-info.png'), dpi=300)
    plt.clf()

if __name__ == '__main__':
    print(getPartitionPoint(os.getcwd(), 'alexnet', True, True, 4))