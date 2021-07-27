import gym
import gym_gazeboros_ac
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

ENV_NAME = 'gazeborosAC-v0'
EPISODE_LEN = 15

# Robot Chase Simulator 2021
# How to use:
# Terminal 1: Launch turtlebot.launch
# Terminal 2: run `python tf_node.py in old_scripts`
# Terminal 3: Launch navigation.launch
# Terminal 4: run this file
#
# * DON'T FORGET TO SOURCE THE WORKSPACE IN EACH FILE <3
# ie: cd .../far_ws && source devel/setup.bash

if __name__ == '__main__':
    print('START Move Test')




    def compute_action_set(orientation_rad):
        pi = np.pi
        numb_tickers = 16
        phase_shift = 2*pi/numb_tickers

        velocity_ratios = [1/(1.6*1.6), 1/1.6, 1] # 1.66 or 1.625 or 1.6

        action_set = []
        action_set.append([0, 0]) # do nothing

        for velocity_ratio in velocity_ratios:

            angle = orientation_rad - phase_shift
            for i in range(3): # 3 is hardcoded, if changed, reorientation & plot will be needed 
               
                # (velocity_ratio*np.cos(angle), velocity_ratio*np.sin(angle))
                action_set.append([velocity_ratio, angle]) # [linear_velocity, angular_velocity]
                angle += phase_shift # TODO was angle += phase_shift
 
        return action_set # 10 actions


    action_set = compute_action_set(0)
    print(action_set)
    mode = 4
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    action = [0.5, 0] # linear_velocity, angular_velocity. from 0 to 1, a % of the max_linear_vel (0.8) & max_angular_vel (1.8)
    # while False:
    while True:
        # env.set_person_mode(mode % 5)
        mode += 1
        state = env.reset()
        env.person.pause()
        # env.person.resume()

        c = 0
        for i in range(1000000):# EPISODE_LEN
            c = c+1
            c = c % 10 
            print(f'c is {c}')
            state, reward, done, _ = env.step(action)
            # dx_dt, dy_dt, da_dt = env.get_system_velocities() # best to see code. (dx_dt, dy_dt, da_dt)
            # print(f'X: {dx_dt} | Y: {dy_dt} | Angular V: {da_dt}')

            # Prints out x y heading position of person
            # person_state = env.get_person_pos()  # [xy[0], xy[1], theta] where theta is orientation
            # print(f'Person state is {person_state}')

            # print(f'State is {state}') # shape is 47

            print(f"Robot state \n\t position is {env.robot.state_['position']} \n\t orientation is {env.robot.state_['orientation']} \n\t velocity lin & angular is {env.robot.state_['velocity']}")
            print(f'Person state\n\t position is {env.person.state_["position"]}\n\t orientation is {env.person.state_["orientation"]}\n\t velocity lin & angular is {env.person.state_["velocity"]}')

            rel_pos = env.get_relative_position(env.person.get_pos(), env.robot)
            distance = np.hypot(rel_pos[0], rel_pos[1])

            print(f'get relative position. person.pos()-robot.pos(): {rel_pos} | with a distance of {distance}')


            rel_heading = env.get_relative_heading_position(env.robot, env.person)[1]
            orientation_rad = np.arctan2(rel_heading[1], rel_heading[0])
            orientation = np.rad2deg(orientation_rad)
            print(f'get relative heading: {rel_heading} | orientation_rad {orientation_rad} | orientation {orientation}')


            action_set = compute_action_set(orientation_rad)
            # print(f'action_set {action_set}')
            action = action_set[c]
            sleep(0.25)
            # sleep(2.00)
            # if done:
            #     break

            # c += 1
    
    print("END")


