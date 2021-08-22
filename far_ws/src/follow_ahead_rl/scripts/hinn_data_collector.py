import random
import numpy as np
import pandas as pd
import os
import pickle
from time import sleep
from collections import deque

import gym
import gym_gazeboros_ac

# Constants
ENV_NAME = 'gazeborosAC-v0'
N_EPISODES = 1000
STEPS_PER_EPISODE = 35
# ADDON_PREV_DATA = True # removed
Human_xyTheta_ordered_triplets = []

def saveData(Human_xyTheta_ordered_triplets=Human_xyTheta_ordered_triplets, ADDON_PREV_DATA=True):
    # if ADDON_PREV_DATA:
    if os.path.isfile(save_local):
        with open(save_local, 'rb') as handle:
            x = pickle.load(handle)
            Human_xyTheta_ordered_triplets.extend(x) # pd.read_csv(save_local).values.tolist()
    else:
        print(f"Warning: Tried to load previous data but files were not found!\nLooked in location {save_local}")

    # save data 
    with open(save_local, 'wb') as handle:
        pickle.dump(Human_xyTheta_ordered_triplets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # _ = pd.DataFrame(Human_xyTheta_ordered_triplets).to_csv(save_local, header=False, index=False)
      

if __name__ == '__main__':
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)

    # save_local_1 = './model_weights/HumanIntentNetwork/Saves/list_of_human_state.csv'
    # save_local_2 = './model_weights/HumanIntentNetwork/Saves/list_of_human_state_next.csv'
    # list_of_human_state = []
    # list_of_human_state_next = []

    save_local= './model_weights/HumanIntentNetwork/Saves/Human_xyTheta_ordered_triplets.pickle'

    action = [0,0] 
    mode = [0,1,2] # 4 # see def path_follower() is gym_gazebros.py
    window_size = 4 # 1 is current, rest are past
    history = deque([0]*window_size, maxlen=window_size)

    for i in range(N_EPISODES):
        print(f"Running episode: {i} of {N_EPISODES}")
        env.set_person_mode(random.choice(mode))   # Cycle through different person modes

        state = env.reset()

        # fill with valid non-0 data
        xyTheta = env.get_person_pos()
        for _ in range(window_size+1):
            history.appendleft(xyTheta)

        for ii in range(STEPS_PER_EPISODE):

            sleep(0.5)
            xyTheta = env.get_person_pos()
            history.appendleft(xyTheta)

            action = [0,0]
            # action = [xyTheta[0], xyTheta[1]]
            state, reward, done, _ = env.step(action) # i doubt this even matters... 

            # print(f'history as list is\n{list(history)}') # we are saving a moving window. it's not efficient, but that shouldnt matter
            Human_xyTheta_ordered_triplets.append(list(history))

            ''' past code
                state, reward, done, _ = env.step(action)
                human_state = list(state)
                list_of_human_state.append(human_state) # current x,y,theta AND WITH it's history    to next x,y, theta 
                sleep(0.1)
                xy = env.get_person_pos()
                next_state = [xy[0], xy[1], xy[2]]
                list_of_human_state_next.append(next_state)
            '''
            if env.fallen:
                state = env.reset()
                break
        
        if i % 20 == 0:
            saveData()
            Human_xyTheta_ordered_triplets = []
            print("Done Saving...")

    env.close()
    saveData()
    print("Done Saving...\nEND")

    ''' past code
        if os.path.isfile(save_local_1) and os.path.isfile(save_local_2):
            list_of_human_state.extend(pd.read_csv(save_local_1).values.tolist())
            list_of_human_state_next.extend(pd.read_csv(save_local_2).values.tolist())
            _ = pd.DataFrame(list_of_human_state).to_csv(save_local_1, header=False, index=False)
            _ = pd.DataFrame(list_of_human_state_next).to_csv(save_local_2, header=False, index=False)
    '''