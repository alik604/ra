import pickle
import math
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

    # between pose and pose. where pose is position and orientation, and the 2nd pose is the "center"
    def get_relative_pose(pos_goal, orientation_goal, pos_center, orientation_center):
        center_pos = np.asarray(pos_center)
        center_orientation = orientation_center
        
        relative_pos = np.asarray(pos_goal)
        relative_pos2 = np.asarray([relative_pos[0] + math.cos(orientation_goal),
                                    relative_pos[1] + math.sin(orientation_goal)]).T

        # transform the relative to center coordinat
        rotation_matrix = np.array([[np.cos(center_orientation), np.sin(center_orientation)], # TODO Try both with viz. Ali: I think this is a bug. it should be -center_orientation, like in other `rotation_matrix`s
                                    [-np.sin(center_orientation), np.cos(center_orientation)]])
        relative_pos = np.matmul(relative_pos, rotation_matrix)
        relative_pos2 = np.matmul(relative_pos2, rotation_matrix)
        global_pos = np.asarray(relative_pos + center_pos)
        global_pos2 = np.asarray(relative_pos2 + center_pos)
        new_orientation = np.arctan2(global_pos2[1]-global_pos[1], global_pos2[0]-global_pos[0])
        
        return global_pos[0], global_pos[1], new_orientation


    # def compute_action_set(orientation_rad):
    #     pi = np.pi
    #     numb_tickers = 16
    #     phase_shift = 2*pi/numb_tickers

    #     velocity_ratios = [1/(1.6*1.6), 1/1.6, 1] # 1.66 or 1.625 or 1.6

    #     action_set = []
    #     action_set.append([0, 0]) # do nothing

    #     for velocity_ratio in velocity_ratios:

    #         angle = orientation_rad - phase_shift
    #         for i in range(3): # 3 is hardcoded, if changed, reorientation & plot will be needed 
               
    #             # (velocity_ratio*np.cos(angle), velocity_ratio*np.sin(angle))
    #             action_set.append([velocity_ratio, angle]) # [linear_velocity, angular_velocity]
    #             angle += phase_shift # TODO was angle += phase_shift
 
    #     return action_set # 10 actions

    def compute_action_set_from_TEB():
        trajectories = []
        with open('discrete_action_space.pickle', 'rb') as handle:
            x = pickle.load(handle)
            x, y, theta = list(zip(*x))
            for i in range(len(x)):
                # print(f'\t{x[i]}, {y[i]}')
                # plt.plot(x[i], y[i])
                trajectories.extend([[x[i], y[i], theta[i]]])
        return trajectories

    # action_set = compute_action_set(0)
    trajectories = compute_action_set_from_TEB()
    # for i in range(len(trajectories)): # look like the first elem is indeed the first (meaning its not flipped)
        # for ii in range(len(trajectories[i])):
        # print(f'trajectories[i][0] {trajectories[i][0]}\n')
        # print(f' {abs(trajectories[i][0][0]) < abs(trajectories[i][0][-1])}')
    # exit()
    mode = 4
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    action = [0.5, 0] # linear_velocity, angular_velocity. from 0 to 1, a % of the max_linear_vel (0.8) & max_angular_vel (1.8)
    counter = 0
    # while False:
    while True:
        # env.set_person_mode(mode % 5)
        mode += 1
        state = env.reset()
        # env.person.pause() # weird side effect for ending episode (path finished)
        # env.person.resume()
        counter += 1
        # counter = counter % 10 
        print(f'counter is {counter}')       
        for i in range(1):# EPISODE_LEN
                       
            # dx_dt, dy_dt, da_dt = env.get_system_velocities() # best to see code. (dx_dt, dy_dt, da_dt)
            # print(f'X: {dx_dt} | Y: {dy_dt} | Angular V: {da_dt}')

            # Prints out x y heading position of person
            # person_state = env.get_person_pos()  # [xy[0], xy[1], theta] where theta is orientation
            # print(f'Person state is {person_state}')

            # print(f'State is {state}') # shape is 47

            # print(f"Robot state \n\t position is {env.robot.state_['position']} \n\t orientation is {env.robot.state_['orientation']} \n\t velocity lin & angular is {env.robot.state_['velocity']}")
            # print(f'Person state\n\t position is {env.person.state_["position"]}\n\t orientation is {env.person.state_["orientation"]}\n\t velocity lin & angular is {env.person.state_["velocity"]}')

            rel_pos = env.get_relative_position(env.person.get_pos(), env.robot)
            distance = np.hypot(rel_pos[0], rel_pos[1])

            # print(f'get relative position. person.pos()-robot.pos(): {rel_pos} | with a distance of {distance}')


            rel_heading = env.get_relative_heading_position(env.robot, env.person)[1]
            orientation_rad = np.arctan2(rel_heading[1], rel_heading[0])
            orientation = np.rad2deg(orientation_rad)
            # print(f'get relative heading: {rel_heading} | orientation_rad {orientation_rad} | orientation {orientation}')

            recommended_move = np.random.choice(len(trajectories))
            path_to_simulate = trajectories[recommended_move].copy()

            current_robot_pos = env.robot.state_['position']
            print(f'path_to_simulate is {path_to_simulate[:2]}')
            for idx in range(len(path_to_simulate[0])):  # TODO this is wrong
                path_to_simulate[0][idx] += current_robot_pos[0]
                path_to_simulate[1][idx] += current_robot_pos[1]
            path_to_simulate = np.around(path_to_simulate, 2)
            print(f'current_robot_pos is {current_robot_pos}\npath_to_simulate is {path_to_simulate[:2]}')
            # exit()

            #### option a #####         
            # cords = path_to_simulate[-1]
            # action = [cords[0], cords[1]]
            # state_rel_person, reward, done, _ = env.step(action)

            #### option b #####   
            NUMBER_SUB_STEPS = len(path_to_simulate)
            for idx in range(NUMBER_SUB_STEPS):
                robot_state = {}
                x, y, theta = path_to_simulate[0][idx], path_to_simulate[1][idx], path_to_simulate[2][idx]
                last_x, last_y, last_theta = current_robot_pos[0], current_robot_pos[1], env.robot.state_['orientation']
                
                # x, y, theta = get_relative_pose([x, y], theta, [last_x, last_y], last_theta)
                state_rel_person, reward, done, _ = env.step([x, y])

            # state, reward, done, _ = env.step(action)
            # action_set = compute_action_set(orientation_rad)
            # print(f'action_set {action_set}')
            # action = action_set[c]
            sleep(0.50)
            # sleep(2.00)
            # if done:
            #     break

            # c += 1
    
    print("END")


