import numpy as np
import torch
import matplotlib.pyplot as plt
# from parsers import args
from ddpg.model import ReplayBuffer, DDPG, Environment
from ddpg.util import getIPLA
from tqdm import tqdm

class Trainer:
    def __init__(self, device_dict, model, exit_points, model_input_size, acc_thres, cals, bwup_rate, dec_dict,
                 test_dataloader, path, core_num, device, batchsize, root_path):
        n_states = 2
        n_actions = len(device_dict['clients']) * 2
        action_bound = 1.  # 动作的最大值 1.0
        self.env = Environment(device_dict, model, exit_points, model_input_size, acc_thres, cals, bwup_rate, test_dataloader, path, core_num, device)
        # 经验回放池实例化
        self.replay_buffer = ReplayBuffer(capacity=5000)
        self.batchsize = batchsize
        self.dec_dict = dec_dict
        # 模型实例化
        self.agent = DDPG(n_states=n_states,  # 状态数
                          n_hiddens=100,  # 隐含层数
                          n_actions=n_actions,  # 动作数
                          action_bound=action_bound,  # 动作最大值
                          sigma=0.05,  # 高斯噪声
                          actor_lr=0.001,  # 策略网络学习率
                          critic_lr=0.001,  # 价值网络学习率
                          tau=0.001,  # 软更新系数
                          gamma=0.99,  # 折扣因子
                          device=device,
                          root_path=root_path)

    def start(self, emax, tmax):
        exit_lists = self.env.getAllExitSets()
        opt_dec = []
        best_ipias = []
        pbar = tqdm(total=len(exit_lists) * emax * tmax)
        for exit_list in exit_lists:
            opt_dec.append(())
            best_ipla = -0x3fffffff
            pbar.set_description('Training...')
            return_list = []
            mean_return_list = []
            for i in range(emax):  # 迭代10回合
                episode_return = 0  # 累计每条链上的reward
                state = self.env.reset(exit_list)  # 初始时的状态

                for step in range(tmax):
                    # 获取当前状态对应的动作
                    action = self.agent.take_action(state)
                    # 环境更新
                    next_state, reward = self.env.step(action)
                    s, f = self.env.processActionVec(action)
                    self.env.getMaxLocalLatency()
                    ipla = getIPLA(next_state, self.env.Tmax).item()

                    if ipla > best_ipla:
                        opt_dec[-1] = (s, exit_list, f)
                        best_ipla = ipla

                    # 更新经验回放池
                    self.replay_buffer.add(state, action.tolist(), reward, next_state)
                    # 状态更新
                    state = next_state
                    # 累计每一步的reward
                    episode_return += reward.item()

                    # 如果经验池超过容量，开始训练
                    if self.replay_buffer.size() > self.batchsize + 1:
                        # 经验池随机采样batch_size组
                        s, a, r, ns = self.replay_buffer.sample(self.batchsize)
                        # 构造数据集
                        transition_dict = {
                            'states': s,
                            'actions': a,
                            'rewards': r,
                            'next_states': ns,
                        }
                        # 模型训练
                        self.agent.update(transition_dict)
                    pbar.update(1)
                    if len(return_list) > 10:
                        pbar.set_postfix(returns=episode_return, ipla=ipla, mean_returns=np.mean(return_list[-10:]))
                    else:
                        pbar.set_postfix(returns=episode_return, ipla=ipla)
                # 保存每一个回合的回报
                return_list.append(episode_return)
                mean_return_list.append(np.mean(return_list[-10:]))  # 平滑

                # 打印回合信息
                # print(f'iter:{i}, return:{episode_return}, mean_return:{np.mean(return_list[-10:])}')

            best_ipias.append(best_ipla)
            self.replay_buffer.clear()

            # # -------------------------------------- #
            # # 绘图
            # # -------------------------------------- #
            #
            # x_range = list(range(len(self.return_list)))
            #
            # plt.subplot(121)
            # plt.plot(x_range, self.return_list)  # 每个回合return
            # plt.xlabel('episode')
            # plt.ylabel('return')
            # plt.subplot(122)
            # plt.plot(x_range, self.mean_return_list)  # 每回合return均值
            # plt.xlabel('episode')
            # plt.ylabel('mean_return')

        self.dec_dict['info'] = opt_dec[torch.argmax(torch.tensor(best_ipias)).item()]
        self.dec_dict['memory'] = self.env.getMemory(self.dec_dict['info'][1])
        self.dec_dict['FLOPs'] = self.env.getFlops(self.dec_dict['info'][1])