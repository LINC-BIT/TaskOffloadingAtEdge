import torch
from model.resnet import ResNet, BasicBlock
import torch.nn as nn
# exit points start from 1
class ResNetWithBranch(ResNet):
    layer_nums = 7
    def __init__(self, block, num_blocks, branches, num_classes=10):
        super().__init__(block, num_blocks, num_classes)
        self.layers = nn.Sequential(
            self.conv1,
            self.bn1,  # exit point 1 is between conv1 and bn1
            self.relu1,
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            nn.Sequential()
        )
        self.branch = nn.ModuleDict()

        for branch, exit_point in branches:
            self.branch[f'exit_point-{exit_point}'] = branch
        self.branch[f'exit_point-{len(self.layers) - 1}'] = nn.Sequential(
            self.avg_pool2d,
            nn.Flatten(),
            self.linear
        )

    def forward(self, x, exit_point=0):
        if self.training:
            results = []
            for point, layer in enumerate(self.layers):
                exit_point_name = f'exit_point-{point}'
                if exit_point_name in self.branch:
                    x_branch = self.branch[exit_point_name](x)
                    results.append(x_branch.unsqueeze(0))
                x = layer(x)
            return torch.cat(results, dim=0)
        else:
            if f'exit_point-{exit_point}' not in self.branch:
                exit_point = len(self.layers) - 1
            exit_point_name = f'exit_point-{exit_point}'
            for point, layer in enumerate(self.layers):
                if exit_point == point:
                    x = self.branch[exit_point_name](x)
                    break
                x = layer(x)
            return x

def resnet18_branchynet_cifar(num_classes = 100, device='cpu'):
    branch1 = nn.Sequential(
        BasicBlock(64, 64),
        nn.AvgPool2d(kernel_size=4),
        nn.Flatten(),
        nn.Linear(4096, num_classes)
    ).to(device)
    exit_point1 = 3
    branch2 = nn.Sequential(
        BasicBlock(128, 128),
        nn.AvgPool2d(kernel_size=4),
        nn.Flatten(),
        nn.Linear(2048, num_classes)
    ).to(device)
    exit_point2 = 5
    return ResNetWithBranch(BasicBlock, [2, 2, 2, 2], [(branch1, exit_point1), (branch2, exit_point2)], num_classes).to(device)\
        , (exit_point1, exit_point2, ResNetWithBranch.layer_nums)

if __name__ == '__main__':
    x = torch.rand((1, 3, 32, 32))
    model, exit_points = resnet18_branchynet_cifar()
    model.eval()
    x = model(x, exit_points[-1])
    print('a')