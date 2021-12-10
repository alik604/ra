import sys
import numpy as np
import random

"""
define the replay buffer and corresponding algorithms like PER
"""

from Agent_utils import net, linear_schedule, reward_recorder, replay_buffer

import gym
from datetime import datetime
import os

import torch


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