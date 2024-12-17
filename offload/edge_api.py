from net.client import Client
import multiprocessing as mp
import os
import torch
from apscheduler.schedulers.background import BackgroundScheduler
import torch.nn as nn
from inference_utils import warmUP

def start_monitor_client(ip, bandwidth_value):
    monitor_cli = Client(ip=ip, port=9998, bw=bandwidth_value)
    monitor_cli.start()
    # print(f"bandwidth monitor : get bandwidth value : {bandwidth_value.value} MB/s")

def scheduler_for_bandwidth_monitor_edge(ip, interval, bandwidth_value):
    # 创建调度器
    scheduler = BackgroundScheduler(timezone='MST')
    # 每隔 interval 秒就创建一个带宽监视进程 用来获取最新带宽
    scheduler.add_job(start_monitor_client, 'interval', seconds=interval, args=[ip,bandwidth_value])
    scheduler.start()

if __name__ == '__main__':
    ip = '10.1.114.109'
    port = 9999
    bw = mp.Value('d', 0.)
    c = Client(ip, port, bw)
    print(f'客户端开启')
    print(f'ip={ip}, port={port}')
    c.start()
    c.join()
    print('成功连接服务器')
    print(f'BandWidth={bw.value:.3f}MB/s')

    tasks_name = ["alexnet", "vgg", "lenet", "mobilenet"]
    # tasks_list = []
    # for i in range(40):
    #     # task_id = random.randint(0, 3)
    #     task_id = i // 10
    #     tasks_list.append(tasks_name[task_id])

    scheduler_for_bandwidth_monitor_edge(ip=ip, interval=2, bandwidth_value=bw) #定期检测网速
    res = {}

    warmUP(nn.Conv2d(3,3,3), torch.rand((1, 3, 224, 224)))
    for model_type in tasks_name:
        print(''.join(['=' for _ in range(30)]))
        print(f'模型种类:{model_type}')
        x = torch.rand((1, 512, 224, 224))
        o = mp.Array('d', 1000)
        l = mp.Value('d', 0.)
        c = Client(ip, port, bw, model_type, os.getcwd(), monitering=False, is_edge_gpu=False)
        c.setOutput(o)
        c.setInput(x)
        c.setLatencyVar(l)
        c.start()
        c.join()
        # print(f'输出结果:{[i for i in o]}')
        print(f'运行延迟:{l.value:.3f}ms')
        res[model_type] = res.get(model_type, 0) + l.value
        print('成功完成任务')

    print(res)
