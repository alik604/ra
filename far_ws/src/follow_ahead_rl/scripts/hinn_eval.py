from time import sleep
import random
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import gym_gazeboros_ac

from HumanIntentNetwork import HumanIntentNetwork
from distance_heuristic import DistanceHeuristic

TRAIN_ON_ONLY_NEW = True

ENV_NAME = 'gazeborosAC-v0'
RANDOMSEED = 42

N_EPISODES = 100
EPISODE_LEN = 15 # Seconds

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    
    # env.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    model = HumanIntentNetwork(inner=128, input_dim=state_dim, output_dim=3)
    model.load_checkpoint()
    model.to(device)

    dis_heuristic = DistanceHeuristic()

    mode = 0
    for i in range(N_EPISODES):
        env.set_person_mode(mode % 5)
        mode += 1
        state = env.reset()

        # Currently set to run for 15 seconds
        for i in range(EPISODE_LEN * 5):
            state_tensor = torch.Tensor(state).to(device)
            ps_prediction = model.forward(state_tensor)

            action = dis_heuristic.calculate_vector(ps_prediction, [])
            
            env.set_marker_pose([ps_prediction[0],ps_prediction[1]])
            sleep(0.1)
            state, reward, done, _ = env.step(action)

    print("END")