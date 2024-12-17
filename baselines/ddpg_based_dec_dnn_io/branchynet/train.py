import torch
import torch.nn as nn
from torch.optim import Adam, SGD, lr_scheduler
from tqdm import tqdm
from branchynet.util import getAccuracyByEarlyExitPoint

def testClassificationBranchyNet(model, dataloader, exit_points, device='cpu'):
    tested_acc_dict = {}
    pbar = tqdm(total=len(exit_points) * len(dataloader), position=0, leave=True)
    model.eval()
    for i, exit_point in enumerate(exit_points):
        pbar.set_description(f'Testing branch-{i + 1}')
        tested_acc_dict[f'Tested branch-{i + 1} acc'] = getAccuracyByEarlyExitPoint(model, dataloader, exit_point, device, pbar)
    return tested_acc_dict

def trainClassificationBranchyNet(model:nn.Module, exit_points, dataloader, test_dataloader, path, epoch_num=100, epoch_per_test=5, opt='Adam', lr=1e-2, wd=5e-4, device='cpu'):
    assert opt in ['Adam', 'SGD'], '不支持输入的优化器类型'

    optimizer = None
    loss_f = nn.CrossEntropyLoss()
    if opt == 'Adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif opt == 'SGD':
        optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)
    model.train()
    pbar = tqdm(total=epoch_num * len(dataloader), position=0, leave=True)
    acc_dict = {}
    tested_acc_dict = {}
    best_acc = 0.
    for epoch in range(epoch_num):
        pbar.set_description(f'Training epoch {epoch}')
        corrs = None
        trained_num = 0
        model.train()
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            results = model(x)
            weights = torch.ones(results.shape[0]).to(device) / results.shape[0]
            losses = torch.cat([loss_f(res, y).unsqueeze(0) for res in results], dim=0)
            total_loss = torch.dot(losses, weights)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            preds = torch.argmax(results, dim=2)
            corr = torch.cat([torch.sum(pred == y).unsqueeze(0) for pred in preds], dim=0).cpu()
            if corrs is None:
                corrs = corr
            else:
                corrs += corr
            trained_num += x.shape[0]

            for i, corr in enumerate(corrs):
                acc_dict[f'branch-{i+1} acc'] = corr.item() / trained_num
            pbar.update(1)
            pbar.set_postfix(**acc_dict, **tested_acc_dict, loss=total_loss.item())

        if (1 + epoch) % epoch_per_test == 0:
            tested_acc_dict = testClassificationBranchyNet(model, test_dataloader, exit_points, device)
        avg = torch.mean(torch.tensor(list(tested_acc_dict.values())))
        if best_acc < avg:
            torch.save(model.state_dict(), path)
            best_acc = avg
        scheduler.step()