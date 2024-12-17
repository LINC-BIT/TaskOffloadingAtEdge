from inference_utils import getModel, Inference, warmUP
import torch
import torch.nn as nn
import time
from apscheduler.schedulers.background import BackgroundScheduler
import psutil

def start_monitor_client():
    warmUP(nn.Conv2d(3, 3, 3), torch.rand((1, 3, 224, 224)),repeat=5)
    # print(f"bandwidth monitor : get bandwidth value : {bandwidth_value.value} MB/s")

def scheduler_for_bandwidth_monitor_edge(interval):
    # 创建调度器
    scheduler = BackgroundScheduler(timezone='MST')
    # 每隔 interval 秒就创建一个带宽监视进程 用来获取最新带宽
    scheduler.add_job(start_monitor_client, 'interval', seconds=interval)
    scheduler.start()

if __name__ == '__main__':
    mem = psutil.virtual_memory()
    print(1000 - 10 * psutil.cpu_percent(interval=1))
    print(f"Free: {mem.free / 1024 ** 2} MB")
    # ip = '127.0.0.1'
    # scheduler_for_bandwidth_monitor_edge(interval=2)
    # tasks_name = ["alexnet", "vgg", "lenet", "mobilenet"]
    # tasks_list = []
    # for i in range(40):
    #     # task_id = random.randint(0, 3)
    #     task_id = i // 10
    #     tasks_list.append(tasks_name[task_id])
    #
    # is_gpu = False
    # res = {}
    # warmUP(nn.Conv2d(3,3,3), torch.rand((1, 3, 224, 224)))
    # for model_type in tasks_list:
    #     print(model_type)
    #     x = torch.rand((1, 512, 224, 224)).to('cuda' if is_gpu else 'cpu')
    #     model = getModel(model_type, x.shape[1]).to('cuda' if is_gpu else 'cpu')
    #
    #
    #     _, latency = Inference(model, x, is_gpu)
    #     res[model_type] = res.get(model_type, 0) + latency
    #
    #     time.sleep(1)
    #
    # for k,_ in res.items():
    #     res[k] /= 10
    #
    # print(res)

