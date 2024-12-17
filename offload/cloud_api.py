from net.server import Server
from apscheduler.schedulers.background import BackgroundScheduler
def start_monitor_ser(ip):
    monitor_ser = Server(ip=ip, port=9998)
    monitor_ser.start()
    # monitor_ser.join()

def scheduler_for_bandwidth_monitor_cloud(ip, interval):
    # 创建调度器
    scheduler = BackgroundScheduler(timezone='MST')
    scheduler.add_job(start_monitor_ser, 'interval', seconds=interval, args=[ip])
    scheduler.start()

if __name__ == '__main__':
    ip = '10.1.114.109'
    port = 9999
    print(f'服务器开启')
    print(f'ip={ip}, port={port}')
    s = Server(ip, port)
    s.start()
    s.join()
    print('成功连接客户端')

    scheduler_for_bandwidth_monitor_cloud(ip, 2)
    while True:
        print(''.join(['=' for _ in range(20)]))
        s = Server(ip, port, monitering=False)
        s.start()
        s.join()
        print('成功完成客户端任务')