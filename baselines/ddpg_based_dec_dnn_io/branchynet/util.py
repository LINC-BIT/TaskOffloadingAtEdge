from copy import deepcopy
import torch.nn as nn
import torch

def getEarlyExitChain(model, exit_point):
    main_chain = []
    chain = None
    for point, layer in enumerate(model.layers):
        if point == exit_point:
            exit_name = f'exit_point-{point}'
            chain = deepcopy(main_chain)
            branch = deepcopy(model.branch[exit_name])
            chain += [layer for layer in branch]
            chain = nn.Sequential(*chain)
            chain = remove_sequential(chain)
            break
        main_chain.append(layer)
    assert chain is not None, '不存在该退出点'
    return chain

def remove_sequential(model):
    new_model = nn.ModuleList()
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            # 展开 nn.Sequential 层为其包含的子层
            for submodule in module:
                new_model.append(submodule)
        elif isinstance(module, nn.Module):
            new_model.append(module)  # 递归处理子模块
    return nn.Sequential(*new_model)

def getAccuracyByEarlyExitPoint(model, dataloader, exit_point, device, pbar=None):
    tested_num = 0.
    corrs = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            res = model(x, exit_point)
            pred = torch.argmax(res, dim=1)
            corr = torch.sum(pred == y).cpu().item()
            corrs += corr
            tested_num += x.shape[0]
            if pbar is not None:
                pbar.update(1)
    return corrs / tested_num