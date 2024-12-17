import pickle
import socket
import time
import torch
import psutil
import numpy as np
import os
import multiprocessing as mp

def send_data(conn:socket.socket, data, msg='msg', show=False):
    #发送数据长度
    send_x = pickle.dumps(data, protocol=pickle.DEFAULT_PROTOCOL)
    conn.sendall(pickle.dumps(len(send_x), protocol=pickle.DEFAULT_PROTOCOL))
    #接收信息表明对方接收到信息
    conn.recv(40)

    #发送全部数据
    conn.sendall(send_x)
    recv_data = conn.recv(1024).decode()
    if show:
        print(f'接收到{recv_data}，表明成功发送数据{msg}')

def send_short_data(conn, x, msg="msg", show=True): #<1024Bytes
    #向另一方发送比较短的数据 接收数据直接使用get_short_data"""
    send_x = pickle.dumps(x, protocol=pickle.DEFAULT_PROTOCOL)
    conn.sendall(send_x)
    if show:
        print(f"短数据\"{msg}\"已经成功发送")  # 表示对面已收到数据

def send_message(conn, x:str, msg="msg", show=False): #<1024Bytes
    #向另一方发送比较短的数据 接收数据直接使用get_short_data"""
    conn.sendall(x.encode())
    if show:
        print(f"信息\"{msg}\"已经成功发送")  # 表示对面已收到数据

def get_message(conn, show=True):
    s = conn.recv(40).decode()
    if show:
        print(f'信息\"{s}\"已经成功接收')
    return s

def get_data(conn:socket.socket):
    #接收数据长度
    data_len = pickle.loads(conn.recv(1024))
    conn.sendall('success'.encode())

    recv_time = 0.
    tstr = b''
    while True:
        sta = time.perf_counter()
        pack = conn.recv(40960)
        recv_time += (time.perf_counter() - sta) * 1000
        tstr += pack

        if len(tstr) >= data_len:
            break
    data = pickle.loads(tstr)
    conn.sendall('successful'.encode())
    return data, recv_time

def get_short_data(conn):
    #获取短数据
    return pickle.loads(conn.recv(1024))

def close_conn(conn):
    #关闭客户端
    conn.close()

def close_socket(s):
    #关闭服务器
    s.close()

def get_bandwidth(conn):
    data, latency = get_data(conn)
    datasize = torch.numel(data) * 4
    bw = (datasize / 1024 / 1024) / (latency / 1000 + 1e-5)
    return bw

def get_speed(bandwidth, network_type='wifi'):
    """
    根据speed_type获取网络带宽
    :param network_type: 3g lte or wifi
    :param bandwidth 对应的网络速度 3g单位为KB/s lte和wifi单位为MB/s
    :return: 带宽速度 单位：Bpms bytes_per_ms 单位毫秒内可以传输的字节数
    """
    transfer_from_MB_to_B = 1024 * 1024
    transfer_from_KB_to_B = 1024

    if network_type == "3g":
        return bandwidth * transfer_from_KB_to_B
    elif network_type == "lte" or network_type == "wifi":
        return bandwidth * transfer_from_MB_to_B
    else:
        raise RuntimeError(f"目前不支持network type - {network_type}")

def monitorCPUInfo(N = 1000):
    def getFLOPS():
        # 定义矩阵的大小（越大计算越精确）
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)

        # 记录开始时间
        start_time = time.perf_counter()

        # 执行矩阵乘法
        np.dot(A, B)

        # 记录结束时间
        end_time = time.perf_counter()

        # 计算所用时间
        elapsed_time = end_time - start_time

        # 计算浮点运算次数：矩阵乘法的FLOPS计算公式为 2 * N^3
        num_operations = 2 * (N ** 3)
        flops = num_operations / elapsed_time

        return flops

    mem = psutil.virtual_memory()
    latency_threshold = (1000 - 10 * psutil.cpu_percent(interval=1)) / 1000.
    memory_threshold = mem.free
    flops = getFLOPS()
    return latency_threshold, memory_threshold, flops

def cpu_task(core_id):
    os.sched_setaffinity(os.getpid(), {core_id})
    _, _, flops = monitorCPUInfo()
    return flops

def moniterServerCPUInfo(core_num):
    # 绑定到指定核心
    # with mp.Pool(processes=core_num) as pool:
    # results = []
    # for core in range(core_num):
    #     results.append(cpu_task(core))

    _, _, flops = monitorCPUInfo()

    # flops = torch.mean(torch.tensor(results)).item()
    flops /= core_num
    return None, None, flops

def closeDevices(dict):
    for conn in dict['clients']:
        close_conn(conn)

    close_socket(dict['server'])