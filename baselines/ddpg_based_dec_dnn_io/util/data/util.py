import os

def create_dir(path:str):
    list = path.replace('\\', '/').split('/')
    id = -1
    now_path = ''
    while not os.path.exists(path):
        id += 1
        now_path = os.path.join(now_path, list[id])
        if not os.path.exists(now_path):
            os.makedirs(now_path)