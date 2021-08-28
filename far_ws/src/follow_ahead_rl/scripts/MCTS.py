import pickle
import os
import random
import math
import gym
import gym_gazeboros_ac

from time import sleep
from collections import deque

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

from DDQN_Discrete import DeepQNetwork, Agent
np.set_printoptions(linewidth=np.inf)

ENV_NAME = 'gazeborosAC-v0'
PATH_POLY = './model_weights/HumanIntentNetwork/PolynomialRegressor'

WINDOW_SIZE = 4-1 # subject to what PolynomialRegressor is trained on than -1 
# person_history = deque([0]*window_size, maxlen=window_size)
# person_history.appendleft([xyTheta])
# list(person_history)

if os.path.isfile(PATH_POLY):
    REGR = pickle.load(open(PATH_POLY, 'rb'))
else:
    # print(f"[Error!] PolynomialRegressor save not found")
    raise Exception

# TODO remake HIMM with current x,y,theta AND WITH it's history    to next x,y, theta 

def predict_person(state_history):
    # TODO allow predicting N seconds in the furture, by calling a loop... low priority 

    state_history = np.array(state_history).reshape(1, -1) # .flatten()
    # print(f'[predict_person] state_history {state_history}')
    state_history = PolynomialFeatures(degree=2).fit_transform(state_history)     # should be fine, fiting shouldn't be necessary for PolynomialFeatures
    y_pred = REGR.predict(state_history)
    # print(f'y_pred {y_pred.flatten()}')
    return y_pred.flatten().tolist()

# between pose and pose. where pose is position and orientation, and the 2nd pose is the "center"
def get_relative_pose(pos_goal, orientation_goal, pos_center, orientation_center):
    center_pos = np.asarray(pos_center)
    center_orientation = orientation_center
    
    relative_pos = np.asarray(pos_goal)
    relative_pos2 = np.asarray([relative_pos[0] + math.cos(orientation_goal),
                                relative_pos[1] + math.sin(orientation_goal)]).T

    # transform the relative to center coordinat
    rotation_matrix = np.array([[np.cos(center_orientation), np.sin(center_orientation)], # TODO Ali: I think this is a bug. it should be -center_orientation, like in other `rotation_matrix`s
                                [-np.sin(center_orientation), np.cos(center_orientation)]])
    relative_pos = np.matmul(relative_pos, rotation_matrix)
    relative_pos2 = np.matmul(relative_pos2, rotation_matrix)
    global_pos = np.asarray(relative_pos + center_pos)
    global_pos2 = np.asarray(relative_pos2 + center_pos)
    new_orientation = np.arctan2(global_pos2[1]-global_pos[1], global_pos2[0]-global_pos[0])
    
    return global_pos[0], global_pos[1], new_orientation

def MCTS(trajectories, person_history_actual, robot_history_actual, Nodes_to_explore):
    """ MCTS

    Args:
        trajectories (np.array): precomputeed list
        Nodes_to_explore (int): top N actions to consider 
        sum_of_qvals (int, optional): sum of rewards. Defaults to 0.

    Returns:
        int: recommended_move, which of N actions to take; which of the `trajectories` to take
    
    Notes:
        # TODO visusal the path of the robot and the human and reward

        # DONE* sum_of_qvals is naive. mayne we should renormalize or discount
        # the below was implementated
        # 0.4*r1+0.4*r2*d**1+0.4*r3*d**2          // we can just def get_reward(self):
        # 0.4+0.4+0.4 = 1.2 # surely this is better, i would take the step to get 0.4 and recompute
        # 0.2+0.5+0.6 = 1.3

        # 0.4+0.40+0.15 = 1.05 # surely this is better, the last is superior by far
        # 0.4+0.45+0.10 = 1.00
        # instead of jsut the policy output, we coinder the rewards outputast as well.

        # 0.1 , 0.1, 0.15, 0.05
        # renormaizel
        # .15 is 150% better .10 ... 15/.10 = .15
    """


    print(f'\n\n[MCTS]')
    print(f'trajectories: {trajectories}')
    print(f'len(trajectories): {len(trajectories)}')

    # predict person's next move (relative to robot's current pose)
    person_pos = env.person.state_["position"]
    person_theta = env.person.state_["orientation"]
    x, y, theta = get_relative_pose(person_pos, person_theta, env.robot.state_["position"], env.robot.state_["orientation"])
    person_history_actual.appendleft([x, y, theta]) # no loop needed, this function is the "loop"

    person_past_state = list(person_history_actual)
    person_next_state = predict_person(person_past_state)
    # output of predict_person should be relative to robot... 
    # x, y, theta = get_relative_pose([person_next_state[0], person_next_state[1]], person_next_state[2], env.robot.state_["position"], env.robot.state_["orientation"])

    person_history_predicted = person_history_actual.copy() # deque(person_past_state, maxlen=WINDOW_SIZE)
    person_history_predicted.appendleft(person_next_state)
    # print(f'person_next_state = {person_next_state}') # [xy[0], xy[1], state[2]]

    # predict robot's next moves
    robot_pos = env.robot.state_["position"]
    robot_theta = env.robot.state_["orientation"]
    robot_history_actual.appendleft([robot_pos[0], robot_pos[1], robot_theta])
    
    state = env.get_observation_relative_robot() # this could come from main, but perhaps it is best to re-query
    QValues = agent.action_probs(state) # there is no noise... exploration vs exploitation
    idices = np.argsort(QValues)[::-1]  # flip to get largest to smallest
    idices = idices[:Nodes_to_explore]  # select top N
    print(f'QValues:\n{QValues} | sum {np.sum(QValues):.2f}')
    print(f'idices to explore {idices}')

    # Recursively search to choose which of moves to recommend
    rewards = []
    for idx in idices:
        path_to_simulate = trajectories[idx]
        print(f'\n\n\n[call MCTS_recursive from MCTS] path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]}')
        # print(f'trajectories are\n{trajectories}\n\n')
        reward = 1.01 * (QValues[idx] + env.get_reward(simulate=False))
        reward = MCTS_recursive(trajectories.copy(), robot_history_actual.copy(),
                                person_history_predicted.copy(), Nodes_to_explore-1, reward, idx)
        rewards.append(reward)
    best_idx = np.argmax(rewards)
    recommended_move = idices[best_idx]

    return recommended_move


def MCTS_recursive(trajectories, robot_history_predicted, person_history_predicted, Nodes_to_explore, past_rewards, exploring_idx, dTime=0.5):
    """ MCTS_recursive
    Args:

        trajectories (np.array): precomputeed list of moves
            path_to_simulate (np.array): path to take (simulated) to get to the start point
                path_to_simulate[0] is x 
                path_to_simulate[0] is y
        robot_history_predicted (deque): stored as [x, y, theta]
        person_history_predicted (deque): [x, y, theta]. this is from `hinn_data_collector.py` which maintians a history for [xy[0], xy[1], state[2]]
        Nodes_to_explore (int): top N actions to consider 
        past_rewards: past rewards
        exploring_idx (int): debug index of which precomputer traj are we branching from
        dTime (float): dTime in velocity calculations. it is 0.5 because that is what is used to sleep in `hinn_data_collector.py`

    Returns:
        int: recommended_move, which of N actions to take; which of the `trajectories` to take

    """

    # TODO add....  array, orientation = self.get_global_position_orientation([x, y], orientation, self.robot)


    print(f'[start MCTS_recursive] exploring idx: {exploring_idx}\ntrajectories are\n{trajectories}\n\n')
    QValues = []
    states_to_simulate_robot = []
    states_to_simulate_person = []
    robot_pos = robot_history_predicted[0].copy()
    path_to_simulate = trajectories[exploring_idx].copy()
    path_to_simulate = np.around(path_to_simulate, 2)
    print(f'[before] path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]}')

    # // offset path_to_simulate with current robot pos
    for idx in range(len(path_to_simulate[0])):  # TODO this is wrong
        path_to_simulate[0][idx] += robot_pos[0]
        path_to_simulate[1][idx] += robot_pos[1]
    path_to_simulate = np.around(path_to_simulate, 2)
    print(f'[after]  path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]} | has been adjust with x {robot_pos[0]} and y {robot_pos[1]}')
    # print(f'trajectories {list(trajectories)}')
    # print(f'path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]}')


    # // [robot] account for history. since env outputs states based on window of last 10. We also ensure pose is relative to robot.
    robot_hist = list(robot_history_predicted)
    robot_hist.reverse()
    for idx in range(len(robot_hist)-1):
        last_x, last_y, last_theta = robot_hist[idx][0], robot_hist[idx][1], robot_hist[idx][2]
        x, y, theta = robot_hist[idx+1][0], robot_hist[idx+1][1], robot_hist[idx+1][2]
        x, y, theta = get_relative_pose([x,y], theta, [last_x, last_y], last_theta)
        robot_state = {}
        angular_velocity = (theta-last_theta)/dTime
        linear_velocity = np.hypot(x-last_x, y-last_y)/dTime # TODO delta time here is worng, need a elegant way to have it. it's dTime or dTime/NUMBER_SUB_STEPS, depending if MCTS or MCTS_recursive called MCTS_recursive (as `robot_history_predicted` might be "robot_history_predicted" or "robot_history_actual" ) 
        robot_state["velocity"] = (linear_velocity, angular_velocity)
        robot_state["position"] = (x, y)
        robot_state["orientation"] = theta
        states_to_simulate_robot.append(robot_state)

    # TODO why no just the last. that is where we step to...
    # // [robot] account for `path_to_simulate`, our chosen trajectory.
    NUMBER_SUB_STEPS = len(path_to_simulate[0])
    time_step = dTime/NUMBER_SUB_STEPS
    for idx in range(NUMBER_SUB_STEPS):
        robot_state = {}
        x, y, theta = path_to_simulate[0][idx], path_to_simulate[1][idx], path_to_simulate[2][idx]
        last_x, last_y, last_theta = robot_history_predicted[0][0], robot_history_predicted[0][1], robot_history_predicted[0][2]
        
        angular_velocity = (theta-last_theta)/time_step  # fist elem is latest. angular_velocity is dTheta/dTime
        linear_velocity = np.hypot(x-last_x, y-last_y)/time_step 
        x, y, theta = get_relative_pose([x,y], theta, [last_x, last_y], last_theta)

        robot_state["velocity"] = (linear_velocity, angular_velocity)
        robot_state["position"] = (x, y)
        robot_state["orientation"] = theta
        states_to_simulate_robot.append(robot_state)
        robot_history_predicted.appendleft([x, y, theta])
        # print(f'robot_state["position"] {robot_state["position"]}')

    # // [person] predict person's next move. and account for history # add newest to front. flip and build `states_to_simulate_person`. ref for math https://courses.lumenlearning.com/boundless-physics/chapter/quantities-of-rotational-kinematics/
    person_next_state = predict_person(list(person_history_predicted))
    person_history_predicted.appendleft(person_next_state)

    state_ = states_to_simulate_robot[-1]
    pos_state_ = state_['position']
    theta_state_ = state_['orientation']
    person_hist = list(person_history_predicted)
    person_hist.reverse() # oldest to latest
    for idx in range(len(person_hist)-1):
        last_x, last_y, last_theta = person_hist[idx][0], person_hist[idx][1], person_hist[idx][2]
        last_x, last_y, last_theta = get_relative_pose([last_x, last_y], last_theta, pos_state_, theta_state_)
        x, y, theta = person_hist[idx+1][0], person_hist[idx+1][1], person_hist[idx+1][2]
        x, y, theta = get_relative_pose([x,y], theta, pos_state_, theta_state_)
        person_state = {}
        angular_velocity = (theta-last_theta)/dTime
        linear_velocity = np.hypot(x-last_x, y-last_y)/dTime
        person_state["velocity"] = (linear_velocity, angular_velocity)
        person_state["position"] = (x, y)
        person_state["orientation"] = theta
        states_to_simulate_person.append(person_state)
        # print(f'predicted next state of person = {person_state}') # [xy[0], xy[1], state[2]]

    # predict robot's next best moves & select top N
    state = env.get_observation_relative_robot(states_to_simulate_robot, states_to_simulate_person)
    QValues = agent.action_probs(state) # there is no noise... exploration vs exploitation
    idices = np.argsort(QValues)[::-1]  # flip to get largest to smallest
    idices = idices[:Nodes_to_explore]  # select top N
    print(f'QValues:\n{QValues} | sum {np.sum(QValues):.2f}')
    print(f'idices to explore {idices}')

    if Nodes_to_explore == 1:
        print(f'[tail] path_to_simulate: {path_to_simulate}')
        return 0.975*(QValues[idices[0]]*env.get_reward(simulate=True)) + past_rewards
    else:
        # Recursively search
        rewards = []
        print(f'robot_pos was {robot_pos}')
        print(f'robot_pos is now {robot_history_predicted[0]}')
        for idx in idices:
            print(f'\n\n\n[call MCTS_recursive from MCTS_recursive] path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]}')
            # we need both scalers
            current_reward = (0.98*QValues[idx]*env.get_reward(simulate=True)) + (0.99 * past_rewards)
            # print(f'[before recursivly calling MCTS_recursive]\ntrajectories are\n{trajectories}\n\n')
            reward = MCTS_recursive(trajectories.copy(), robot_history_predicted.copy(),
                                    person_history_predicted.copy(), Nodes_to_explore-1, current_reward, exploring_idx=idx)
            rewards.append(reward)
        best_idx = np.argmax(rewards)
        recommended_move = idices[best_idx]

        print(f'[MCTS_recursive] recommended_move is {recommended_move}')
        return recommended_move


if __name__ == '__main__':
    trajectories = []
    with open('discrete_action_space.pickle', 'rb') as handle:
        x = pickle.load(handle)
        x, y, theta = list(zip(*x))
    for i in range(len(x)):
        # print(f'\t{x[i]}, {y[i]}')
        # plt.plot(x[i], y[i])
        trajectories.extend([[x[i], y[i], theta[i]]])
    # plt.show()
    trajectories = trajectories[1:5] # TODO remove 

    # trajectories = trajectories[:3]

    # vect1 =
    # trajectories = [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
    #                 [[2.0, 2.0, 2.0, 2.0], [2.0, 2.0, 2.0, 2.0]],
    #                 [[3.0, 3.0, 3.0, 3.0], [3.0, 3.0, 3.0, 3.0]],
    #                 [[10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0]]]

    # print(f'trajectories: {trajectories}')
    # print(f'numb of trajectories is: {len(trajectories)}')
    # trajectories = np.array(trajectories, dtype=object)
    # for i in range(len(trajectories)):
    #   for ii in range(len(trajectories[i])):
    #     trajectories[i][ii] = np.array(trajectories[i][ii])
    # print(trajectories[i][ii])
    print(f'trajectories: {trajectories}')

    person_history_actual = deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE)
    robot_history_actual = deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE)

    n_actions = len(trajectories)
    observation_shape = 47
    agent = Agent(gamma=0.99, epsilon=0.99, batch_size=128, n_actions=n_actions, eps_end=0.01,
              input_dims=[observation_shape], lr=0.02, eps_dec=5e-4, ALIs_over_training=2, file_label = "DDQN_MCTS") # changed from eps_dec=5e-4
    # agent.save_models()
    agent.load_models()

    print('START Test')
    N_GAMES = 100
    MODES = [0,1,2]
    best_score = -100
    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    # linear_velocity, angular_velocity. from 0 to 1, a % of the max_linear_vel (0.8) & max_angular_vel (1.8)

    for game in range(N_GAMES):
        
        state_rel_person = env.reset()
        env.set_agent(0)
        score = 0 
        done = False

        person_pos = env.person.state_["position"]
        person_theta = env.person.state_["orientation"]
        for _ in range(WINDOW_SIZE):
            person_history_actual.appendleft([person_pos[0], person_pos[1], person_theta])

        robot_pos = env.robot.state_["position"]
        robot_theta = env.robot.state_["orientation"]
        for _ in range(WINDOW_SIZE):
            robot_history_actual.appendleft([robot_pos[0], robot_pos[1], robot_theta])

        # mode = random.choice(MODES)
        # print(f"Running game: {game} of {N_GAMES} | Person Mode {mode}")
        # env.set_person_mode(mode)
         
        observation = env.get_observation_relative_robot()
        # env.person.pause()
        # env.person.resume()
        
        while not done:
            # print(f'state:\n{state}')
            recommended_move = MCTS(trajectories.copy(), person_history_actual, robot_history_actual, Nodes_to_explore=3)
            # TODO take recommended_move
            print(f'in main loop recommended_move is {recommended_move}')

            # take action
            # for cords in trajectories[recommended_move]: # TODO confirm this is right
            cords = trajectories[recommended_move][-1]
            action = [cords[0], cords[1]]
            state_rel_person, reward, done, _ = env.step(action)
            observation_ = env.get_observation_relative_robot()

            agent.store_transition(observation, recommended_move, reward, observation_, done)
            agent.learn()
            observation = observation_
            score +=reward
            

        # if score > best_score:
        #     best_score = score
        #     agent.save_models()
        if i % 30 == 0:
            agent.save_models() # TODO necessary evil
    
    agent.save_models()
    print("DONE")
    env.close()
    exit(0)
