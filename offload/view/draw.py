from glob import glob
import json
import matplotlib.pyplot as plt
import os

def drawPics(y, names, fig_name, x_name, y_name, savedir):
    plt.figure()
    plt.title(fig_name)
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    for yy, name in zip(y, names):
        x = list(range(len(yy)))
        plt.plot(x, yy, label=name)
        print(f'{fig_name}_{name}:{yy[-1]}')
    plt.legend()
    plt.savefig(os.path.join(savedir, fig_name.lower().replace(' ', '_') + '.png'), bbox_inches='tight')

if __name__ == '__main__':
    # paths = glob('./data/*.json')
    paths = glob('./data2/*.json')
    acc_list = []
    l_list = []
    mem_list = []
    flops_list = []
    names = []
    for path in paths:
        tpath = path.replace('\\', '/')
        acc_array, l_array, mem_array, flops_array = json.load(open(tpath, 'r'))
        acc_list.append(acc_array)
        l_list.append(l_array)
        n_mem_array = []
        sum_mem = 0.
        for i, mem in enumerate(mem_array):
            sum_mem += mem
            n_mem_array.append(sum_mem / (i + 1))
        mem_list.append(n_mem_array)
        n_flops_array = []
        sum_flops = 0.
        for i, flops in enumerate(flops_array):
            sum_flops += flops
            n_flops_array.append(sum_flops / (i + 1))
        flops_list.append(n_flops_array)

        name = tpath.split('/')[-1].replace('.json', '')
        names.append(name)

    drawPics(acc_list, names, 'Accuracy Curve', 'Task ID', 'Accuracy', './fig')
    drawPics(l_list, names, 'Mean Latency Curve', 'Task ID', 'Mean Latency (ms)', './fig')
    drawPics(mem_list, names, 'Memory Curve', 'Task ID', 'Memory (MB)', './fig')
    drawPics(flops_list, names, 'FLOPs Curve', 'Task ID', 'mFLOPs', './fig')
