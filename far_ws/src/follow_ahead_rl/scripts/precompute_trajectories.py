"""
see the function `build_action_discrete_action_space()` in gym_gazeboros_ac.py

rostopic pub /move_base_simple/goal_0 geometry_msgs/PoseStamped  "header:
  seq: 0
  stamp:
    secs: 0
    nsecs: 0
  frame_id: 'tb3_0/base_link'
pose:
  position:
    x: 10.0
    y: 10.0
    z: 0.0
  orientation:
    x: 0.0
    y: 0.0
    z: 0.0
    w: 2.0"

[log out]
rostopic echo /move_base_node_0/current_goal
"""

import gym
import gym_gazeboros_ac
import numpy as np
import matplotlib.pyplot as plt
from time import sleep
import random, pickle



ENV_NAME = 'gazeborosAC-v0'



if __name__ == '__main__':


    # l1 = [[1,2,3], [1,2,3], [1,2,3]]
    # l2 = [[9,8,7], [9,8,7], [9,8,7]]

    # x = zip(l1,l2)
    # print(type(x))
    # x = tuple(x)
    # print(type(x))
    # x, y = list(zip(*x))
    # print(x)
    # print(y)
    # #########

    # with open('action_discrete_action_space.pickle', 'rb') as handle:
    #     x = pickle.load(handle)
    # x, y = list(zip(*x))

    # print(f'[in action_test.py]')
    # for i in range(len(x)):
    #   print(f'[{i}/{len(x)-1}]')
    #   print(f'\t{x[i]}, {y[i]}')
    #   print(f'\t{tuple(zip(x[i], y[i]))}')
    #   plt.plot(x[i], y[i])
    # plt.show()
    # exit()
    
    print('START Move Test')

    mode = 4
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    action = [0.0, 0.0] # linear_velocity, angular_velocity. from 0 to 1, a % of the max_linear_vel (0.8) & max_angular_vel (1.8)
    # while False:
    while True:
        # env.set_person_mode(mode % 5)
        mode += 1
        state = env.reset()
        # env.person.pause() # weird side effect for ending episode (path finished)
        # env.person.resume()

        for i in range(1000000):# EPISODE_LEN
            state, reward, done, _ = env.step(action)

            x = env.build_action_discrete_action_space()
            x, y = list(zip(*x))

            print(f'[in action_test.py]')
            # for i in range(len(x)):
            #   print(f'{x[i]}, {y[i]}')
            #   print(f'\t{tuple(zip(x[i], y[i]))}')
            state, reward, done, _ = env.step(action)


            sleep(3.00)
            print("END")
            env.close()
            exit(1)
            if done:
                break    
            print("END")


