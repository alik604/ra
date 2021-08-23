import pickle
import os
import random
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

def predict_person(state):

    # TODO allow predicting N seconds in the furture, by calling a loop... low prioirity 

    # print(f'state.shape is {state.shape}')
    state = state.reshape(1, -1) #
    # print(f'state is {state}')
    # TODO should be fine, fiting shouldn't be necessary for PolynomialFeatures
    state = PolynomialFeatures(degree=2).fit_transform(state)
    # print(f'state.shape is {state.shape}')
    y_pred = REGR.predict(state)
    # print(f'y_pred {y_pred.flatten()}')
    return y_pred.flatten().tolist()


def MCTS(trajectories, person_history_actual, Nodes_to_explore):
    """ MCTS

    Args:
        trajectories (np.array): precomputeed list
        Nodes_to_explore (int): top N actions to consider 
        sum_of_qvals (int, optional): sum of rewards. Defaults to 0.

    Returns:
        int: recommended_move, which of N actions to take; which of the `trajectories` to take
    
    Notes:
        # TODO the trajectories list is being changed somewhere and somehow.....
        # TODO add person pos, and robot velocity
        # save and pass the person pos like `robot_pos`?
        # TODO deal with orientation when simulating

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

    # predict person's next move
    person_pos = env.person.state_["position"]
    person_theta = env.person.state_["orientation"]
    person_history_actual.appendleft([person_pos[0], person_pos[1], person_theta])

    person_past_state = list(person_history_actual)
    person_next_state = predict_person(person_past_state)

    person_history_predicted = deque(person_past_state, maxlen=WINDOW_SIZE)
    person_history_predicted.appendleft(person_next_state)
    # print(f'person_next_state = {person_next_state}') # [xy[0], xy[1], state[2]]

    # predict robot's next moves
    state = env.get_observation_relative_robot()
    QValues = agent.action_probs(state) # there is no noise... exploration vs exploitation
    idices = np.argsort(QValues)[::-1]  # flip to get largest to smallest
    idices = idices[:Nodes_to_explore]  # select top N
    print(f'QValues:\n{QValues} | sum {np.sum(QValues):.2f}')
    print(f'idices to explore {idices}')

    # Recursively search to choose which of moves to recommend
    rewards = []
    robot_pos = env.robot.state_["position"]  # np.array([1, 1])
    for idx in idices:
        path_to_simulate = trajectories[idx]
        print(f'\n\n\n[call MCTS_recursive from MCTS] path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]}')
        print(f'trajectories are\n{trajectories}\n\n')
        reward = 1.01 * (QValues[idx] + env.get_reward(simulate=False))
        reward = MCTS_recursive(robot_pos, trajectories.copy(),
                                person_history_predicted.copy(), Nodes_to_explore-1, reward, idx)
        rewards.append(reward)
    best_idx = np.argmax(rewards)
    recommended_move = idices[best_idx]

    print(f'recommended_move is {recommended_move}')
    return recommended_move


def MCTS_recursive(robot_pos, trajectories, person_history_predicted, Nodes_to_explore, past_rewards, exploring_idx, dTime=0.5):
    """ MCTS_recursive
    Args:
        path_to_simulate (np.array): path to take (simulated) to get to the start point
          path_to_simulate[0] is x 
          path_to_simulate[0] is y
        robot_pos: x, y
        trajectories (np.array): precomputeed list of moves
        person_history_predicted (deque): [x, y, theta]. this is from `hinn_data_collector.py` which maintians a history for [xy[0], xy[1], state[2]]
        Nodes_to_explore (int): top N actions to consider 
        past_rewards: past rewards
        exploring_idx (int): debug index of which precomputer traj are we branching from
        dTime (float): dTime in velocity calculations. it is 0.5 because that is what is used to sleep in `hinn_data_collector.py`

    Returns:
        int: recommended_move, which of N actions to take; which of the `trajectories` to take

    """
    print(f'[start MCTS_recursive]\ntrajectories are\n{trajectories}\n\n')
    QValues = []
    states_to_simulate_robot = []
    states_to_simulate_person = []
    path_to_simulate = trajectories[exploring_idx].copy()
    print(f'[before] path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]}')
    # offset path_to_simulate
    for idx in range(len(path_to_simulate[0])):  # TODO this is wrong
        path_to_simulate[0][idx] += robot_pos[0]
        path_to_simulate[1][idx] += robot_pos[1]
    print(f'[after]  path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]} | has been adjust with x {robot_pos[0]} and y {robot_pos[1]}')
    print(f'[MCTS_recursive] exploring idx: {exploring_idx}')
    # print(f'trajectories {list(trajectories)}')

    # build robot states to simulated
    # TODO why no just the last. that is where we step to... 
    # print(f'path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]}')
    print(f'path_to_simulate theta: {path_to_simulate[2]}')
    for idx in range(len(path_to_simulate[0])):
        robot_state = {}
        robot_state["velocity"] = (0.9, 0) # TODO figure this out
        robot_state["position"] = (path_to_simulate[0][idx], path_to_simulate[1][idx])
        robot_state["orientation"] = path_to_simulate[2][idx]
        states_to_simulate_robot.append(robot_state)
        # print(f'robot_state["position"] {robot_state["position"]}')

    # TODO from x,y we can used arcTan to get the oriantation 

    # predict person's next move
    person_next_state = predict_person(list(person_history_predicted))
    person_history_predicted.appendleft(person_next_state)
    person_state = {}

    # https://courses.lumenlearning.com/boundless-physics/chapter/quantities-of-rotational-kinematics/
    # TODO cordinate frame  see get_relative_heading_position and line `math.hypot(rel_person[0], rel_person[1])`
    _history = person_history_predicted
    x = person_next_state[0]
    y = person_next_state[1]
    angular_velocity  = (_history[0][2]-_history[1][2])/dTime  # fist elem is latest. angular_velocity is dTheta/dTime
    # linear_velocity = np.hypot(_history[0][0]-_history[1][0], _history[0][1]-_history[1][1])/dTime   # from my notes during out meeting I have: sqrt(x^2 + y^2)/dTime. might have meant sqrt((x_1 - x_2)^2 + (y_1 - y_2)^2)/dTime
    linear_velocity = np.hypot(x,y)*angular_velocity # linear_velocity is r*angular_velocity
    person_state["velocity"] = (linear_velocity, angular_velocity)
    person_state["position"] = (x, y)
    person_state["orientation"] = person_next_state[2]
    states_to_simulate_person.append(person_state)
    # print(f'predicted next state of person = {person_next_state}') # [xy[0], xy[1], state[2]]

    # predict person's next move & select top N moves
    state = env.get_observation_relative_robot(states_to_simulate=states_to_simulate_robot, states_to_simulate_person=states_to_simulate_person)
    QValues = agent.action_probs(state) # there is no noise... exploration vs exploitation
    idices = np.argsort(QValues)[::-1]  # flip to get largest to smallest
    idices = idices[:Nodes_to_explore]  # select top N
    print(f'QValues:\n{QValues} | sum {np.sum(QValues):.2f}')
    print(f'idices to explore {idices}')

    if Nodes_to_explore == 1:
        print(f'[tail] path_to_simulate: {path_to_simulate}')
        return 0.975*(QValues[idices[0]]*env.get_reward(simulate=False)) + past_rewards
    else:
        # Recursively search
        rewards = []
        print(f'robot_pos was {robot_pos}')
        robot_pos = np.array([path_to_simulate[0][-1], path_to_simulate[1][-1]])
        print(f'robot_pos is now {robot_pos}')
        # TODO calcualte person_pos for next timestep. this is hard :(  https://math.stackexchange.com/questions/2430809/how-to-determine-x-y-position-from-point-p-based-on-time-velocity-and-rate-of-t
        for idx in idices:
            print(f'\n\n\n[call MCTS_recursive from MCTS_recursive] path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]}')
            # we need both scalers
            current_reward = (0.98*QValues[idx]*env.get_reward(simulate=False)) + (0.99 * past_rewards)
            print(f'[before recursivly calling MCTS_recursive]\ntrajectories are\n{trajectories}\n\n')
            reward = MCTS_recursive(robot_pos, trajectories.copy(),
                                    person_history_predicted.copy(), Nodes_to_explore-1, current_reward, exploring_idx=idx)
            rewards.append(reward)
        best_idx = np.argmax(rewards)
        recommended_move = idices[best_idx]

        print(f'[MCTS_recursive] recommended_move is {recommended_move}')
        return recommended_move


if __name__ == '__main__':
    trajectories = []
    with open('action_discrete_action_space.pickle', 'rb') as handle:
        x = pickle.load(handle)
        x, y, theta = list(zip(*x))
    for i in range(len(x)):
        # print(f'\t{x[i]}, {y[i]}')
        # plt.plot(x[i], y[i])
        trajectories.extend([[x[i], y[i], theta[i]]])
    # plt.show()
    trajectories = trajectories[1:]

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

    n_actions = len(trajectories)
    observation_shape = 47
    agent = Agent(gamma=0.99, epsilon=0.99, batch_size=128, n_actions=n_actions, eps_end=0.01,
              input_dims=[observation_shape], lr=0.02, eps_dec=5e-4, ALIs_over_training=2, file_label = "DDQN_MCTS") # changed from eps_dec=5e-4
    # agent.save_models()
    agent.load_models()

    print('START Test')
    N_GAMES = 1
    MODES = [0,1,2]
    best_score = -100
    env = gym.make(ENV_NAME).unwrapped
    
    # linear_velocity, angular_velocity. from 0 to 1, a % of the max_linear_vel (0.8) & max_angular_vel (1.8)

    action = [0.0, 0.0]
    for game in range(N_GAMES):
        state_rel_person = env.reset()
        person_history_actual = deque([0]*WINDOW_SIZE, maxlen=WINDOW_SIZE)
        person_pos = env.person.state_["position"]
        person_theta = env.person.state_["orientation"]
        for _ in WINDOW_SIZE:
            person_history_actual.appendleft([person_pos[0], person_pos[1], person_theta])
        # mode = random.choice(MODES)
        # print(f"Running game: {game} of {N_GAMES} | Person Mode {mode}")
        # env.set_person_mode(mode)
        
        score=0 
        env.set_agent(0)
        observation = env.get_observation_relative_robot()
        # env.person.pause()
        # env.person.resume()
        EPISODE_LEN = 6
        for i in range(EPISODE_LEN):  
            # print(f'state:\n{state}')
            recommended_move = MCTS(trajectories.copy(), person_history_actual, Nodes_to_explore=3)
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

            # sleep(2.00)
        # if score > best_score:
        #     best_score = score
        #     agent.save_models()
        if i % 30 == 0:
            agent.save_models() # TODO necessary evil
    print("DONE")
    env.close()
    exit(0)
        #     if done:
        #         print("DONE")
        #         break
        #     print("END")
        # env.close()
