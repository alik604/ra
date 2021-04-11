""" if you change `init_pos_robot` in `gym_gazeboros_ac.py`, then do not have the code `if done:    break`.
 As it will be an instant done. This is why I have `while max_itr > 0`
 rather then it being an infinitly running loop with being `done` as the exit condition"""

from time import sleep
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import gym
import gym_gazeboros_ac

from HumanIntentNetwork import HumanIntentNetwork

TRAIN_ON_ONLY_NEW = False

ENV_NAME = 'gazeborosAC-v0'
RANDOMSEED = 42

EPISODES = 100  # 2000     # Simulations
STEPS_PER_EPI = 20
EPOCHS = 10  # 1000     # Training
BATCH_SIZE = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    env.seed(RANDOMSEED)
    torch.manual_seed(RANDOMSEED)
    np.random.seed(RANDOMSEED)

    state_dim = 23  # env.observation_space.shape[0] # not updated
    action_dim = env.action_space.shape[0]
    print(f'State_dim:  {state_dim}')
    print(f'Action_dim: {action_dim}')

    save_local_1 = './model_weights/HumanIntentNetwork/list_of_human_state.csv'
    save_local_2 = './model_weights/HumanIntentNetwork/list_of_human_state_next.csv'

    if TRAIN_ON_ONLY_NEW:
        list_of_human_state = []
        list_of_human_state_next = []
    else:
        list_of_human_state = pd.read_csv(save_local_1).values.tolist()
        list_of_human_state_next = pd.read_csv(save_local_2).values.tolist()

    for i in range(EPISODES):
        # env.set_obstacle_pos("obstacle_box", 0.5, 0, 0)
        state = env.reset()

        # Prints out x y position of person
        # print(f"person pose = {env.get_person_pos()}") # human state x,y,z
        # print(state)
        max_itr = STEPS_PER_EPI
        while max_itr > 0:
            max_itr -= 1
            # [random.uniform(-1, 1), random.uniform(-1, 1)]
            action = [-50, 50]
            print(action)

            # state[2] is oriantation.
            state, reward, done, _ = env.step(action)
            # if done:    break
            human_state = list(state)
            list_of_human_state.append(human_state)
            # print(f"person pose: {human_state}")

            # sleep(0.1)
            # np.random()

            state, reward, done, _ = env.step(action)
            # if done:    break # will need to discard non-parallel end

            xy = env.get_person_pos()
            next_state = [xy[0], xy[1], state[2]]
            list_of_human_state_next.append(next_state)
            # print(f"person pose: {human_state_next}")

            # Prints out system velocities
            # print(f"system_velocities = {env.get_system_velocities()}")

        if i % 1 == 0:  # i > 0 and
            print(f'\n\n\n It has been {i} Simulations \n\n\n')
    env.close()

    print(
        f'Before: {len(list_of_human_state)} | {len(list_of_human_state_next)}')
    # discard non-parallel end
    cuttoff = min(len(list_of_human_state), len(list_of_human_state_next))
    list_of_human_state = list_of_human_state[:cuttoff]
    list_of_human_state_next = list_of_human_state_next[:cuttoff]

    if TRAIN_ON_ONLY_NEW:
        # deep copy and have parallel
        cuttoff = len(list_of_human_state_next)
        COPY_list_of_human_state_ = list_of_human_state.copy()
        COPY_list_of_human_state_next_ = list_of_human_state_next.copy()

        # extend copy with saved data
        # COPY_list_of_human_state_.extend(pd.read_csv(save_local_1).values.tolist())
        # COPY_list_of_human_state_next_.extend(pd.read_csv(save_local_2).values.tolist())

        # save data
        _ = pd.DataFrame(COPY_list_of_human_state_).to_csv(
            save_local_1, header=False, index=False)
        _ = pd.DataFrame(COPY_list_of_human_state_next_).to_csv(
            save_local_2, header=False, index=False)
    else:
        # save data
        _ = pd.DataFrame(list_of_human_state).to_csv(
            save_local_1, header=False, index=False)
        _ = pd.DataFrame(list_of_human_state_next).to_csv(
            save_local_2, header=False, index=False)

    print(
        f'After: {len(list_of_human_state)} | {len(list_of_human_state_next)}')

    model = HumanIntentNetwork(inner=128, input_dim=23, output_dim=3)
    model.load_checkpoint()
    model.to(device)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam((model.parameters()), lr=1e-3)

    # TODO shuffle data, in a parallel manner

    list_of_human_state = torch.Tensor(list_of_human_state).to(device)
    list_of_human_state_next = torch.Tensor(
        list_of_human_state_next).to(device)
    # print(list_of_human_state.size(0))

    losses = []
    for epoch in range(EPOCHS):
        summer = 0
        for i in range(0, list_of_human_state.size(0), BATCH_SIZE):
            # print(f'{i} to {i+BATCH_SIZE}')

            target = list_of_human_state_next[i: i+BATCH_SIZE]
            input_batch = list_of_human_state[i: i+BATCH_SIZE]
            pred = model.forward(input_batch)

            optimizer.zero_grad()
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            summer += loss.item()

            # print(f'pred {pred}')
            # print(f'target {target}')

        if epoch % 100 == 0:
            model.save_checkpoint()
        losses.append(summer)
        print(f'Epoch {epoch} | Loss_sum {summer:.4f}')

    model.save_checkpoint()
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss of (possibly) Pre-Trained model')
    plt.show()

    print("END")
