import gym
import gym_gazeboros_ac
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import random, pickle
ENV_NAME = 'gazeborosAC-v0'

def MCTS(trajectories, Nodes_to_explore, sum_of_rewards=0):
  # TODO adjust trajectories to your current location
  # TODO trajectories might not be a nested line (final case), so that will case a issue
  QValues = []
  print(f'trajectories: {trajectories}')
  print(f'len(trajectories): {len(trajectories)}')
  isTail = False
  for idx, trajectory in enumerate(trajectories):
    print(f'..trajectory {trajectory}')
    print(f" \t{trajectory[0]} \n\t{trajectory[1]}\n")
    states = []

    # // lazy overload for tail case
    # if isinstance(trajectory[0], np.float64):
    #   print(f'[setting] trajectory = trajectories')
    #   trajectory = trajectories

    if len(trajectories) == 2:
       trajectory = trajectories
       isTail=True
    for i in range(len(trajectory[0])):
      state = {} 
      state["velocity"] = env.robot.state_["velocity"]# = (1.0, 0) # TODO
      state["position"] = (trajectory[0][i], trajectory[1][i])
      state["orientation"] = 3 #env.robot.state_["orientation"] # = 0  TODO
      states.append(state)
      break # only 1 

    state = env.get_observation_relative_robot(states)
    # TODO get Q value here
    QValues.append(random.randint(0, 5))
    # print(f'obs:\n{state}')

  # select top N moves
  idices = np.argsort(QValues)[::-1] # sort
  idices = idices[:Nodes_to_explore] # select top N
  # print(f'trajectories {trajectories}')
  # print(f'idices {idices}')

  trajectories = trajectories[idices]

  if len(trajectories) == 1 or isTail:
    print(f'[tail] trajectories: {trajectories}')
    return QValues[0]  # TODO
  else:
    # Recursively search
    rewards = [] 
    for idx in idices:
      reward = MCTS(trajectories[idx], Nodes_to_explore-1, sum_of_rewards+QValues[idx])
      rewards.append(reward)
    best_idx = np.argmax(rewards)
    recommended_move = idices[best_idx]

  return recommended_move
    
    
      
    
      

if __name__ == '__main__':
    trajectories = []
    with open('action_discrete_action_space.pickle', 'rb') as handle:
      x = pickle.load(handle)
      x, y = list(zip(*x))
      for i in range(len(x)):
        # print(f'\t{x[i]}, {y[i]}')
        plt.plot(x[i], y[i])
        trajectories.extend([[x[i], y[i]]])
      # plt.show()
      trajectories = trajectories[1:]
      trajectories = trajectories[:3]
      print(f'numb of trajectories is: {len(trajectories)}')
      trajectories = np.array(trajectories)
    
    print('START Move Test')
    mode = 4
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    action = [0.0, 0.0] # linear_velocity, angular_velocity. from 0 to 1, a % of the max_linear_vel (0.8) & max_angular_vel (1.8)
    while True:
        # env.set_person_mode(mode % 5)
        mode += 1
        state = env.reset()
        # env.person.pause()
        # env.person.resume()

        for i in range(1):# EPISODE_LEN
            state, reward, done, _ = env.step(action)

            print(f'state:\n{state}')
            MCTS(trajectories, Nodes_to_explore=3)


            # sleep(1000.00)

            if done:
                print("DONE")
                break    
            print("END")
            exit(0)
        env.close()


