import sys
import numpy as np
import random

"""
define the replay buffer and corresponding algorithms like PER
"""

<<<<<<< HEAD

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os

import gym
=======
from Agent_utils import net, linear_schedule, reward_recorder, replay_buffer

import gym
from datetime import datetime
import os

import torch
>>>>>>> MCTS


def get_args():
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument('--env-name', type=str, default='CartPole-v1', help='the environment name')

    parse.add_argument('--gamma', type=float, default=0.99, help='the discount factor of RL')
    parse.add_argument('--batch-size', type=int, default=64, help='the batch size of updating')
    parse.add_argument('--lr', type=float, default=1e-4, help='learning rate of the algorithm')
    parse.add_argument('--buffer-size', type=int, default=30000, help='the size of the buffer')
    parse.add_argument('--cuda', action='store_true', default=False, help='if use the gpu')
    parse.add_argument('--init-ratio', type=float, default=0.60, help='the initial exploration ratio')
    parse.add_argument('--exploration-fraction', type=float, default=0.8, help='decide which fraction of steps to do the exploration... like a exploration step size')
    parse.add_argument('--final-ratio', type=float, default=0.01, help='the final exploration ratio')
    parse.add_argument('--n-games', type=int, default=int(500), help='the total games to train network')
    parse.add_argument('--learning-starts', type=int, default=200, help='the Episodes start learning. perhaps we wish to fill the replay buffer')
    parse.add_argument('--train-freq', type=int, default=1, help='the frequency to update the network')
    parse.add_argument('--save-dir', type=str, default='saved_models/', help='the folder to save models')

    args = parse.parse_args()

    return args

<<<<<<< HEAD
=======

>>>>>>> MCTS
class dqn_agent:
    def __init__(self):
        self.args = get_args()
        self.env = gym.make(self.args.env_name) # gym.make('LunarLander-v2')

        # define the network
        state_dim = self.env.reset().shape[0] #self.env.observation_space
        num_actions = self.env.action_space.n
        self.net = net(state_dim=state_dim, hidden=64, hidden2=32, hidden3=8, num_actions=num_actions)
        self.target_net = net(state_dim=state_dim, hidden=64, hidden2=32, hidden3=8, num_actions=num_actions)

        # create the folder to save the models
        if not os.path.exists(self.args.save_dir):
            os.mkdir(self.args.save_dir)
        # set the environment folder
        self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        else:
            self.load_net(path=self.model_path + '/model.pt')

        # make sure the target net has the same weights as the network
        self.target_net.load_state_dict(self.net.state_dict())

        if self.args.cuda:
            self.net.cuda()
            self.target_net.cuda()

        # define the optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.args.lr)
        # define the replay memory
        self.buffer = replay_buffer(self.args.buffer_size)
        # define the linear schedule of the exploration
        self.exploration_schedule = linear_schedule(int(self.args.n_games * self.args.exploration_fraction), self.args.final_ratio, self.args.init_ratio)

<<<<<<< HEAD
    # start to do the training
    def learn(self):
        # the episode reward
        episode_reward = reward_recorder()
        obs = np.array(self.env.reset())
        td_loss = 0
        for game in range(self.args.n_games):
            done = False
            while not done:
                self.env.render()
                explore_eps = self.exploration_schedule.get_value(game)
                with torch.no_grad():
                    obs_tensor = self._to_tensors(obs)
                    action_value = self.net(obs_tensor)
                # select actions
                action = self.select_actions(action_value, explore_eps)

                # execute actions
                obs_, reward, done, _ = self.env.step(action)
                obs_ = np.array(obs_)
                # append the samples
                self.buffer.add(obs, action, reward, obs_, float(done))
                obs = obs_
                # add the rewards
                episode_reward.add_rewards(reward)
                if game % self.args.train_freq == 0:
                    # start to sample the samples from the replay buffer
                    batch_samples = self.buffer.sample(self.args.batch_size)
                    td_loss = self._update_network(batch_samples)

                if game % 150 == 0:
                    # update the target network
                    self.target_net.load_state_dict(self.net.state_dict())

                if done and episode_reward.num_episodes % 10 == 0:
                    print('[{}], Episode: {}, Rewards: {:.0f}, Loss: {:.3f}, Explore_eps {:.3f}'.format(datetime.now(), episode_reward.num_episodes, episode_reward.value, td_loss, explore_eps))
                    torch.save(self.net.state_dict(), self.model_path + '/model.pt')

                if done:
                    obs = np.array(self.env.reset())
                    # start new episode to store rewards
                    episode_reward.start_new_episode() # history last 1 episode

        self.env.close()

=======
>>>>>>> MCTS
    # update the network
    def _update_network(self, samples):
        obses, actions, rewards, obses_next, dones = samples
        # convert the data to tensor
        obses = self._to_tensors(obses)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        obses_next = self._to_tensors(obses_next)
        dones = torch.tensor(1 - dones, dtype=torch.float32).unsqueeze(-1)
        # convert into gpu
        if self.args.cuda:
            actions = actions.cuda()
            rewards = rewards.cuda()
            dones = dones.cuda()
        # calculate the target value
        with torch.no_grad():
            q_value_ = self.net(obses_next)
            action_max_idx = torch.argmax(q_value_, dim=1, keepdim=True)
            target_action_value = self.target_net(obses_next)
            target_action_max_value = target_action_value.gather(1, action_max_idx)

        # target
        expected_value = rewards + self.args.gamma * target_action_max_value * dones
        # get the real q value
        action_value = self.net(obses)
        real_value = action_value.gather(1, actions)
        loss = (expected_value - real_value).pow(2).mean()
        # start to update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    # get tensors
    def _to_tensors(self, obs):
        obs = torch.tensor(obs, dtype=torch.float32)
        if self.args.cuda:
            obs = obs.cuda()
        return obs

    # Load weights to net
    def load_net(self, path):
        print(f'Loading weights from disk onto self.net')
        self.net.load_state_dict(torch.load(path))

    def select_actions(self, action_value, explore_eps):
        action_value = action_value.cpu().numpy().squeeze()
        # print(f'action_value {action_value}')
        action = np.argmax(action_value) if random.random() > explore_eps else np.random.randint(action_value.shape[0])
        return action

<<<<<<< HEAD
    def select_actions_boltzmann(self, action_value, temperature=1.0): # alt temperature 0.5
        # TODO is the the same as softmax?
        action_value = action_value.cpu().numpy().squeeze()

        # select actions
        weights = np.exp(action_value.values / temperature)
        distribution = {action: weights[action] / np.sum(weights) for action in num_actions} # self.num_actions
        return np.random.choice(list(distribution.keys()), 1, p=np.array(list(distribution.values())))[0]

        # action = np.argmax(action_value) if random.random() > explore_eps else np.random.randint(action_value.shape[0])
        # return action

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

    def forward(self, inputs):

        x = F.relu(self.linear1(inputs))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.linear3(x))

        # get the action value
        action_fc = F.relu(self.action_fc(x))
        action_value = self.action_value(action_fc)
        # get the state value
        state_value_fc = F.relu(self.state_value_fc(x))
        state_value = self.state_value(state_value_fc)
        # action value mean
        action_value_mean = torch.mean(action_value) # , dim=1, keepdim=True)
        agvantage_function = action_value - action_value_mean # Q - V
        # Q = V + A
        action_value_out = state_value + agvantage_function
        return action_value_out

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

agent = dqn_agent()
agent.learn()
=======
    def select_actions_boltzmann(self, action_value, n_actions, temperature=1.0, return_action_and_prob=False): # alt temperature 0.5
        # TODO is the the same as softmax?
        # action_value = action_value.cpu().numpy().squeeze()
        # select actions
        # weights = np.exp(action_value / temperature)
        # distribution = {action: weights[action] / np.sum(n_actions) for action in n_actions} # self.num_actions
        # action = np.random.choice(list(distribution.keys()), 1, p=np.array(list(distribution.values())))[0]

        # print(action_value)
        softmax = torch.softmax(action_value, 0)
        distribution = softmax.numpy()
        action = np.random.choice(n_actions, 1, p=distribution)[0]
        # print(f'probs {softmax.numpy()}')
        # print(f'action {action}')
        # print(f'prov of action selected {distribution[action]}')

        if return_action_and_prob:
            return action, distribution[action]
        else:
            return action



# start to do the training
def learn(agent):
    # the episode reward
    episode_reward = reward_recorder()
    obs = np.array(agent.env.reset())
    td_loss = 0
    for game in range(agent.args.n_games):
        done = False
        while not done:
            agent.env.render()
            explore_eps = agent.exploration_schedule.get_value(game)
            with torch.no_grad():
                obs_tensor = agent._to_tensors(obs)
                action_value = agent.net(obs_tensor)
            # select actions
            action = agent.select_actions(action_value, explore_eps)

            # execute actions
            obs_, reward, done, _ = agent.env.step(action)
            obs_ = np.array(obs_)
            # append the samples
            agent.buffer.add(obs, action, reward, obs_, float(done))
            obs = obs_
            # add the rewards
            episode_reward.add_rewards(reward)
            if game % agent.args.train_freq == 0:
                # start to sample the samples from the replay buffer
                batch_samples = agent.buffer.sample(agent.args.batch_size)
                td_loss = agent._update_network(batch_samples)

            if game % 150 == 0:
                # update the target network
                agent.target_net.load_state_dict(agent.net.state_dict())

            if done and episode_reward.num_episodes % 10 == 0:
                print('[{}], Episode: {}, Rewards: {:.0f}, Loss: {:.3f}, Explore_eps {:.3f}'.format(datetime.now(), episode_reward.num_episodes, episode_reward.value, td_loss, explore_eps))
                torch.save(agent.net.state_dict(), agent.model_path + '/model.pt')

            if done:
                obs = np.array(agent.env.reset())
                # start new episode to store rewards
                episode_reward.start_new_episode()  # history last 1 episode

    agent.env.close()

# agent = dqn_agent()
# learn(agent)
>>>>>>> MCTS
