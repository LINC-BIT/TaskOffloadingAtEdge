import torch

def getIPLA(state, Tmax):
    acc, lantency = state
    return -(1. / Tmax) * lantency + acc

def getFixedF(y, self_core_num):
    # 按比例分配并初步取整
    proportions = torch.softmax(y, dim=0) * self_core_num
    f = torch.floor(proportions).tolist()  # 初步向下取整
    f = [max(1, int(v)) for v in f]  # 确保每项至少为 1

    # 调整使总和等于 self.core_num
    current_sum = sum(f)
    diff = self_core_num - current_sum

    # 根据剩余差值调整（优先补最大值对应的项）
    if diff > 0:
        for _ in range(diff):
            max_idx = torch.argmax(torch.tensor(f)).item()
            f[max_idx] += 1
    elif diff < 0:
        for _ in range(-diff):
            min_idx = torch.argmin(torch.tensor(f)).item()
            f[min_idx] -= 1

    # 确保调整后没有 0
    assert all(v > 0 for v in f), "每一项必须大于 0"
    assert sum(f) == self_core_num, "总和必须等于 self_core_num"

    return f
