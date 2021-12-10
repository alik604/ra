import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class replay_buffer:
    def __init__(self, memory_size):
        self.storage = []
        self.memory_size = memory_size
        self.next_idx = 0
        self.only_once = True

    # add the samples
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storage):
            self.storage.append(data)
        else:
            self.storage[self.next_idx] = data
        # get the next idx
        tmp = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.memory_size
        if self.only_once and tmp+1 != self.next_idx:
            print(f'replay_buffer is full')
            self.only_once = False

    # encode samples
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storage[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)

class reward_recorder: # history last 1 episode
    def __init__(self, history_length=100):
        self.history_length = history_length
        # the empty buffer to store rewards
        self.buffer = [0.0]
        self._episode_length = 1

    # add rewards
    def add_rewards(self, reward):
        self.buffer[-1] += reward

    # start new episode
    def start_new_episode(self):
        # if len(self.buffer) >= self.history_length:
        #     self.buffer.pop(0)
        # # append new one
        # self.buffer.append(0.0)
        # self._episode_length += 1

        self._episode_length += 1
        self.buffer = [0.0]

    @property
    def mean(self):
        return np.mean(self.buffer)

    @property
    def value(self):
        return self.buffer[-1]

    @property
    def num_episodes(self):
        # get the length of total episodes
        return self._episode_length

class linear_schedule:
    def __init__(self, total_timesteps, final_ratio, init_ratio=1.0):
        self.total_timesteps = total_timesteps
        self.final_ratio = final_ratio
        self.init_ratio = init_ratio

    def get_value(self, timestep):
        frac = min(float(timestep) / self.total_timesteps, 1.0)
        return self.init_ratio - frac * (self.init_ratio - self.final_ratio)

class net(nn.Module):
    def __init__(self, state_dim, hidden, hidden2, hidden3, num_actions):
        super(net, self).__init__()
        # define the network
        self.linear1 = nn.Linear(state_dim, hidden)
        self.linear2 = nn.Linear(hidden, hidden2)
        # self.linear3 = nn.Linear(hidden2, hidden2)

        self.action_fc = nn.Linear(hidden2, hidden3)
        self.action_value = nn.Linear(hidden3, num_actions)

        self.state_value_fc = nn.Linear(hidden2, hidden3)
        self.state_value = nn.Linear(hidden3, 1)
        self.state_value_output = 0
    def forward(self, inputs):

        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))

        # get the action value
        action_fc = F.relu(self.action_fc(x))
        action_value = self.action_value(action_fc)
        # get the state value
        state_value_fc = F.relu(self.state_value_fc(x))
        self.state_value_output = self.state_value(state_value_fc)
        # action value mean
        action_value_mean = torch.mean(action_value) # , dim=1, keepdim=True)

        agvantage_function = action_value - action_value_mean # Q - V
        # Q = V + A
        action_value_out = self.state_value_output + agvantage_function
        return action_value_out
