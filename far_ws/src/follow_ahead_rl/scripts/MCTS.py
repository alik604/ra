import random, pickle, os
from time import sleep
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import gym
import gym_gazeboros_ac

np.set_printoptions(linewidth=np.inf)

ENV_NAME = 'gazeborosAC-v0'
PATH_Poly = './model_weights/HumanIntentNetwork/PolynomialRegressor'

if os.path.isfile(PATH_Poly):
  REGR = pickle.load(open(PATH_Poly, 'rb'))
else:
  # print(f"[Error!] PolynomialRegressor save not found")
  raise Exception

def predict_person(state):
  # print(f'state.shape is {state.shape}')
  state = state.reshape(1, -1)
  # print(f'state is {state}')
  state = PolynomialFeatures(degree=2).fit_transform(state) # TODO should be fine, fiting shouldn't be necessary for PolynomialFeatures
  # print(f'state.shape is {state.shape}')
  y_pred = REGR.predict(state)
  # print(f'y_pred {y_pred.flatten()}')
  return y_pred.flatten()

def MCTS(trajectories, Nodes_to_explore):
  """ MCTS

  Args:
      trajectories (np.array): precomputeed list
      Nodes_to_explore (int): top N actions to consider 
      sum_of_qvals (int, optional): sum of rewards. Defaults to 0.

  Returns:
      int: recommended_move, which of N actions to take; which of the `trajectories` to take
  """
  # TODO the trajectories list is being changed somewhere and somehow.....
  # TODO add person pos, and robot velocity 
   # save and pass the person pos like `robot_pos`?
  # TODO deal with orientation when simulating

  # TODO add Q-network()

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


  print(f'\n\n[MCTS]')
  print(f'trajectories: {trajectories}')
  print(f'len(trajectories): {len(trajectories)}')

  # get person's move
  person_state_3value = None
  state = env.get_observation_relative_robot(HINN=True) # TODO it is not relative to person, because that's not what the traning data was. 
  person_state_3value = predict_person(state)
  # print(f'person_state_3value = {person_state_3value}') # [xy[0], xy[1], state[2]]
  
  state = env.get_observation_relative_robot()
  # TODO get Q(state))
  QValues = np.random.rand(len(trajectories))
  QValues /= np.sum(QValues)
  print(f'QValues:\n{QValues} | sum {np.sum(QValues):.2f}')

  # select top N moves
  idices = np.argsort(QValues)[::-1] # sort
  idices = idices[:Nodes_to_explore] # select top N
  print(f'idices to explore: {idices}')

  # Recursively search
  rewards = []
  robot_pos = env.robot.state_["position"] # np.array([1, 1])
  person_pos = env.person.state_["position"]
  for idx in idices:
    path_to_simulate = trajectories[idx]
    print(f'\n\n\n[call MCTS_recursive from MCTS] path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]}')
    reward = 1.01 * (QValues[idx] + env.get_reward(simulate=False))
    reward = MCTS_recursive(robot_pos, person_pos, trajectories, person_state_3value, Nodes_to_explore-1, reward, idx)
    rewards.append(reward)
  best_idx = np.argmax(rewards)
  recommended_move = idices[best_idx]

  print(f'recommended_move is {recommended_move}')
  return recommended_move
    
def MCTS_recursive(robot_pos, person_pos,  trajectories, person_state_3value, Nodes_to_explore, past_rewards, exploring_idx):
  """ MCTS_recursive
  Args:
      path_to_simulate (np.array): path to take (simulated) to get to the start point
        path_to_simulate[0] is x 
        path_to_simulate[0] is y
      robot_pos: x, y
      trajectories (np.array): precomputeed list of moves
      person_state_3value (np.array): [x, y, theta]. this is from `hinn_data_collector.py` which has [xy[0], xy[1], state[2]]
      Nodes_to_explore (int): top N actions to consider 
      past_rewards: past rewards
      exploring_idx (int): debug index of which precomputer traj are we branching from

  Returns:
      int: recommended_move, which of N actions to take; which of the `trajectories` to take

  """  
  QValues = []
  states_to_simulate = []
  states_to_simulate_person = []
  path_to_simulate = trajectories[exploring_idx].copy()

  # offset path_to_simulate 
  for idx in range(len(path_to_simulate[0])): # TODO this is wrong 
    path_to_simulate[0][idx] += robot_pos[0]
    path_to_simulate[1][idx] += robot_pos[1]
  print(f'path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]} | has been adjust with x {robot_pos[0]} and y {robot_pos[1]}')

  print(f'[MCTS_recursive] exploring idx: {exploring_idx}')
  # print(f'trajectories {list(trajectories)}')
  
  # // robot 
  # print(f'path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]}')
  print(f'path_to_simulate theta: {path_to_simulate[2]}')
  for idx in range(len(path_to_simulate[0])):
    state = {} 
    state["velocity"] = (0.9, 0) # env.robot.state_["velocity"]# = (1.0, 0) # TODO figure this out
    state["position"] = (path_to_simulate[0][idx], path_to_simulate[1][idx])
    state["orientation"] = path_to_simulate[2][idx]  
    states_to_simulate.append(state)
    # print(f'state["position"] {state["position"]}')

  # // person
  #  person_state_3value [xy[0], xy[1], state[2]]

  print(f'person_state_3value is {person_state_3value}')
  state = {} 
  state["velocity"] = (person_state_3value[0], person_state_3value[1]) 
  state["position"] = (person_pos[0], person_pos[1]) 
  state["orientation"] = person_state_3value[2] #env.robot.state_["orientation"] # = 0 
  states_to_simulate_person.append(state)

  state = env.get_observation_relative_robot(states_to_simulate=states_to_simulate, states_to_simulate_person=states_to_simulate_person, HINN=True)
  person_state_3value = predict_person(state) # update person_state_3value, will be used for recursion 
  # print(f'person_state_3value = {person_state_3value}') # [xy[0], xy[1], state[2]]

  state = env.get_observation_relative_robot(states_to_simulate=states_to_simulate, states_to_simulate_person=states_to_simulate_person) # different 
  # TODO get Q value here
  QValues = np.random.rand(len(trajectories))
  QValues /= np.sum(QValues)
  print(f'QValues:\n{QValues} | sum {np.sum(QValues):.2f}')

  # select top N moves
  idices = np.argsort(QValues)[::-1] # flip to get largest to smallest  
  idices = idices[:Nodes_to_explore] # select top N
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
    person_pos = env.person.state_["position"]
    for idx in idices:
      print(f'\n\n\n[call MCTS_recursive from MCTS] path_to_simulate x: {path_to_simulate[0]} | y: {path_to_simulate[1]}')
      reward = (0.98*QValues[idx]*env.get_reward(simulate=False)) + (0.99 * past_rewards) # we need both scalers
      reward = MCTS_recursive(robot_pos, person_pos, trajectories, person_state_3value, Nodes_to_explore-1, reward, exploring_idx=idx)
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

      for i in range(10):# EPISODE_LEN
          

          # print(f'state:\n{state}')
          recommended_move = MCTS(trajectories, Nodes_to_explore=3)
          # TODO take recommended_move
          print(f'in main loop recommended_move is {recommended_move}')

          # take action
          # for cords in trajectories[recommended_move]: # TODO confirm this is right
          cords = trajectories[recommended_move][-1]
          action = [cords[0], cords[1]]
          state, reward, done, _ = env.step(action)

          # sleep(2.00)
      print("DONE")
      env.close()
      exit(0)
        #     if done:
        #         print("DONE")
        #         break    
        #     print("END")
        # env.close()


