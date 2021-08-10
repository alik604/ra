import gym
import gym_gazeboros_ac
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import random, pickle
ENV_NAME = 'gazeborosAC-v0'

def MCTS(trajectories):
  for trajectory in trajectories:
    print(f" \t{trajectory[0]} \n\t{trajectory[1]}\n")
    states = []
    for i in range(len(trajectory[0])):
      state = {} 
      state["velocity"] = env.robot.state_["velocity"]# = (1.0, 0) # TODO
      state["position"] = (trajectory[0][i], trajectory[1][i])
      state["orientation"] = env.robot.state_["orientation"] # = 0  TODO
      states.append(state)

    state = env.get_observation_relative_robot(states)

    print(f'obs:\n{state}')
    return
    
      

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

        for i in range(5):# EPISODE_LEN
            state, reward, done, _ = env.step(action)

            print(f'state:\n{state}')
            MCTS(trajectories)


            sleep(3.00)
            if done:
                break    
            print("END")
        env.close()


