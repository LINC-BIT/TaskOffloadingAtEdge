import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import collections
import numpy as np
from branchynet.util import getAccuracyByEarlyExitPoint
import itertools
from ddpg.util import getIPLA, getFixedF

class ReplayBuffer:
    def __init__(self, capacity):  # 经验池的最大容量
        # 创建一个队列，先进先出
        self.buffer = collections.deque(maxlen=capacity)

    # 在队列中添加数据
    def add(self, state, action, reward, next_state):
        # 以list类型保存
        self.buffer.append((state, action, reward, next_state))

    def clear(self):
        self.buffer.clear()

    # 在队列中随机取样batch_size组数据
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        # 将数据集拆分开来
        state, action, reward, next_state = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state)

    # 测量当前时刻的队列长度
    def size(self):
        return len(self.buffer)

class QValueNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512):
        super().__init__()
        self.l1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = self.l1(x)
        x = self.l2(x)
        q_value = self.output(x)
        return q_value

class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=512, action_bound=1.):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

    def forward(self, state):
        x = self.l1(state)
        x = self.l2(x)
        raw_action = self.output(x)
        action = torch.tanh(raw_action)
        # action = self.action_bound * raw_action
        return action

class Environment:
    def __init__(self, devices_dict, model, exit_points, model_input_size, acc_thres, cals, bwup_rate, test_dataloader, path, core_num, device='cpu'):
        self.clients = devices_dict['clients']
        self.server = devices_dict['server']
        self.model = model
        self.model_input_size = model_input_size
        self.accuracy_thres = acc_thres
        self.exit_points = exit_points
        self.test_dataloader = test_dataloader
        self.server_device = device
        self.core_num = core_num
        profile_path = os.path.join(path, 'profiles.yaml')
        self.profile = yaml.load(open(profile_path, 'r'), yaml.FullLoader)

        # TODO:
        # self.cals = getCalAbilities(server, clients, self.server_device)
        self.cals = cals
        self.bwup_rate = bwup_rate
        self.upload_rates = []
        self.download_rates = []
        self.mapping = {}
        self.outputs = {}
        self.accs = {}
        self.flops = {}
        self.ks = {}
        self.exit_point_list = None
        self.last_state = None
        self.k = None
        self.acc_mean = None
        self.Tmax = None
        self.memory = {}

        self._processData()

    def _getGoodExitPointsSets(self, e_set):
        local_num = len(self.clients)
        results = list(itertools.product(e_set, repeat=local_num))
        return results

    def _processData(self):
        for k, v in self.profile.items():
            self.mapping[v['exit point']] = k
            self.outputs[k] = [np.prod(np.array(v[f'layer-{i}']['output'])) * 4 / (1024 ** 2) for i in range(v['chain length'])]
            self.flops[k] = [v[f'layer-{i}']['FLOPs'] for i in range(v['chain length'])]
            self.memory[k] = [v[f'layer-{i}']['param'] * 4 for i in range(v['chain length'])]
            self.accs[k] = v['acc']
            self.ks[k] = v['chain length']
            self.input = np.prod(np.array(v['layer-0']['input']))

    def getMemory(self, exit_points):
        memories = []
        for ep in exit_points:
            memories.append(sum(self.memory[self.mapping[ep]]))
        return memories

    def getFlops(self, exit_points):
        flops = []
        for ep in exit_points:
            flops.append(sum(self.flops[self.mapping[ep]]))
        return flops

    def processActionVec(self, action):
        x = torch.softmax(action[:len(self.exit_point_list)], dim=0)
        y = action[len(self.exit_point_list):]

        s = torch.floor(self.k * x).to(dtype=torch.int).tolist()
        f = getFixedF(y, self.core_num)

        return s, f


    def step(self, action):
        reward = 0.
        s, f = self.processActionVec(action)

        avg_latency = 0.
        for i, p in enumerate(s):
            latency = 0.
            exit_point = self.exit_point_list[i]
            branch_name = self.mapping[exit_point]
            for id, o in enumerate(self.outputs[branch_name]):
                if id + 1 < p:
                    latency += self.flops[branch_name][id] / self.cals[i + 1]
                if p == 0:
                    latency += self.input / self.cals[i + 1]
                if p == id + 1:
                    latency += o / self.bwup_rate[i]
                if id + 1 > p:
                    latency += self.flops[branch_name][id] / (f[i] * self.cals[0])
            avg_latency += latency * self.model_input_size[0]
        avg_latency /= len(s)
        state = [self.acc_mean, avg_latency] #t+1
        if self.last_state is not None:
            reward = getIPLA(self.last_state, self.Tmax)
        self.last_state = state
        return state, reward

    def reset(self, exit_point_list):
        self.exit_point_list = exit_point_list
        self.acc_mean = torch.mean(torch.tensor([self.accs[self.mapping[ep]] for ep in self.exit_point_list]))
        self.k = torch.tensor([self.ks[self.mapping[ep]] for ep in self.exit_point_list])
        action = torch.rand(len(exit_point_list) * 2)
        self.getMaxLocalLatency()
        state, _ = self.step(action)
        return state

    def getAllExitSets(self):
        e_set = []
        for exit_point in self.exit_points[::-1]:
            if self.accs[self.mapping[exit_point]] >= self.accuracy_thres:
                e_set.append(exit_point)
        return self._getGoodExitPointsSets(e_set)

    def getMaxLocalLatency(self):
        max_latency = 0.
        for i, exit_point in enumerate(self.exit_point_list):
            latency = 0.
            branch_name = self.mapping[exit_point]
            for id in range(self.k[i]):
                latency += self.flops[branch_name][id] / self.cals[i + 1]
            max_latency = max(max_latency, latency)
        self.Tmax = max_latency * self.model_input_size[0]

class DDPG:
    def __init__(self, n_states, n_hiddens, n_actions, action_bound,
                 sigma, actor_lr, critic_lr, tau, gamma, device, root_path):
        # 策略网络--训练
        self.actor = PolicyNet(n_states, n_actions, n_hiddens, action_bound).to(device)
        # 价值网络--训练
        self.critic = QValueNet(n_states, n_actions, n_hiddens).to(device)

        self.path = os.path.join(root_path, 'ddpg.pth')
        if os.path.exists(self.path):
            state_dicts = torch.load(self.path, map_location=device)
            self.actor.load_state_dict(state_dicts['actor'])
            self.critic.load_state_dict(state_dicts['critic'])

        # 策略网络--目标
        self.target_actor = PolicyNet(n_states, n_actions, n_hiddens, action_bound).to(device)
        # 价值网络--目标
        self.target_critic = QValueNet(n_states, n_actions, n_hiddens).to(device)
        # 初始化价值网络的参数，两个价值网络的参数相同
        self.target_critic.load_state_dict(self.critic.state_dict())
        # 初始化策略网络的参数，两个策略网络的参数相同
        self.target_actor.load_state_dict(self.actor.state_dict())

        # 策略网络的优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        # 价值网络的优化器
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # 属性分配
        self.gamma = gamma  # 折扣因子
        self.sigma = sigma  # 高斯噪声的标准差，均值设为0
        self.tau = tau  # 目标网络的软更新参数
        self.n_actions = n_actions
        self.device = device

    def save_model(self):
        dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }
        torch.save(dict, self.path)
    # 动作选择
    def take_action(self, state):
        # 维度变换 list[n_states]-->tensor[1,n_states]-->gpu
        state = torch.tensor(state, dtype=torch.float).view(1, -1).to(self.device)
        # 策略网络计算出当前状态下的动作价值 [1,n_states]-->[1,1]-->int
        action = self.actor(state)[0]
        # 给动作添加噪声，增加搜索
        action = action + self.sigma * torch.randn(self.n_actions)
        return action

    # 软更新, 意思是每次learn的时候更新部分参数
    def soft_update(self, net, target_net):
        # 获取训练网络和目标网络需要更新的参数
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            # 训练网络的参数更新要综合考虑目标网络和训练网络
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    # 训练
    def update(self, transition_dict):
        # 从训练集中取出数据
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)  # [b,n_states]
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).to(self.device)  # [b,1]
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(
            self.device)  # [b,1]
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)  # [b,next_states]
        # dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)  # [b,1]

        # 价值目标网络获取下一时刻的动作[b,n_states]-->[b,n_actors]
        next_action = self.target_actor(next_states)
        # 策略目标网络获取下一时刻状态选出的动作价值 [b,n_states+n_actions]-->[b,1]
        next_q_values = self.target_critic(next_states, next_action)
        # 当前时刻的动作价值的目标值 [b,1]
        q_targets = rewards + self.gamma * next_q_values

        # 当前时刻动作价值的预测值 [b,n_states+n_actions]-->[b,1]
        q_values = self.critic(states, actions)

        # 预测值和目标值之间的均方差损失
        critic_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # 价值网络梯度
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # 当前状态的每个动作的价值 [b, n_actions]
        actor_q_values = self.actor(states)
        # 当前状态选出的动作价值 [b,1]
        score = self.critic(states, actor_q_values)
        # 计算损失
        actor_loss = -torch.mean(score)
        # 策略网络梯度
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新策略网络的参数
        self.soft_update(self.actor, self.target_actor)
        # 软更新价值网络的参数
        self.soft_update(self.critic, self.target_critic)
