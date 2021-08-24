import gym
import gym_gazeboros_ac
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

import time, os 
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, 
            n_actions, name, chkpt_dir='./model_weights/DDQN_Discrete', file_label="null"):
        super(DeepQNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = f'{chkpt_dir}'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+f'_{file_label}')
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc2_5 = nn.Linear(self.fc2_dims, int(self.fc2_dims/2))
        self.fc3 = nn.Linear(int(self.fc2_dims/2), self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        # self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        print(self.device)
        self.to(self.device)

    def forward(self, state):
        state = state.float()
        # print(state.dtype)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_5(x))
        actions = self.fc3(x)

        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        try:
            self.load_state_dict(T.load(self.checkpoint_file))
        except FileNotFoundError:
            print('... loading failed. Will non destructively save random weights and reload, to prevent reoccurrence...')
            T.save(self.state_dict(), self.checkpoint_file)
            self.load_checkpoint()
            raise Exception(f'\n\nIf you have weights, correct the file name... else rerun\n[debug]Checkpoint_file is {self.checkpoint_file}')
            


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
            max_mem_size=100000, eps_end=0.05, eps_dec=5e-4, ALIs_over_training=2, file_label="null"):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.iter_cntr = 0
        self.replace_target = 30
        self.ALIs_over_training = ALIs_over_training

        self.Q_eval = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=512, fc2_dims=256, name="Q_eval", file_label=file_label)
        self.Q_next = DeepQNetwork(lr, n_actions=n_actions, input_dims=input_dims,
                                    fc1_dims=512, fc2_dims=256, name="Q_next", file_label=file_label) # 64 ,64, if not updating pramas

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        
    def save_models(self):
        self.Q_eval.save_checkpoint()
        self.Q_next.save_checkpoint()

    def load_models(self):
        self.Q_eval.load_checkpoint()
        self.Q_next.load_checkpoint()


    def store_transition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def action_probs(self, observation):
        state = T.tensor([observation]).to(self.Q_eval.device)
        # actions = self.Q_eval.forward(state).detach().numpy()[0]
        # actions -= actions.min() # TODO recheck should be `+`. or it should be (x - min(x))/ (max(x) - min(x)). but this works. weird...
        # actions /=actions.sum()
        # actions = np.random.choice(actions, Nodes_to_explore_greedy, p=actions, replace=False)

        actions = self.Q_eval.forward(state)
        actions = F.softmax(actions.detach()).numpy()[0]
        # print(f'[action_probs] actions is {actions} | {np.sum(actions)}')
        return actions

    def learn(self):
        if self.mem_cntr < self.batch_size: #maybe self.batch_size*2... IDK about this 
            return

        max_mem = min(self.mem_cntr, self.mem_size)

        # replace=False means dont give duplicates. max_mem isnt mutated
        batch = np.random.choice(max_mem, self.batch_size, replace=False) # todo decrease and force train on last 3 
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        
        # N = 2 if self.iter_cntr > self.batch_size else 1 # maybe use self.mem_cntr
        for i in range(self.ALIs_over_training): # Ali over training 
            self.Q_eval.optimizer.zero_grad()
            q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
            q_next = self.Q_eval.forward(new_state_batch)
            q_next[terminal_batch] = 0.0

            q_target = reward_batch + self.gamma*T.max(q_next, dim=1)[0]

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            loss.backward()
            self.Q_eval.optimizer.step()

            self.iter_cntr += 1
            
            if self.iter_cntr % self.replace_target == 0:
                self.Q_next.load_state_dict(self.Q_eval.state_dict())
                
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        
        # This isn't my code. IDK why we dont optimize Q_next, however, I trust the author (youtube: machine learning with Phil). 
        # This was because the two networks are different... IDK how to update the Q_next network


if __name__ == '__main__':
    # from https://raw.githubusercontent.com/philtabor/Youtube-Code-Repository/master/ReinforcementLearning/DeepQLearning/simple_dqn_torch_2020.py and https://github.com/philtabor/Youtube-Code-Repository/blob/master/ReinforcementLearning/DeepQLearning/main_torch_dqn_lunar_lander_2020.py


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

    print('START Move Test')
    ENV_NAME = 'gazeborosAC-v0'

    n_actions = 10
    observation_shape = 47
    # agent = Agent(gamma=0.99, epsilon=0.35, batch_size=64, n_actions=n_actions, eps_end=0.01,
    #           input_dims=[observation_shape], lr=0.001, eps_dec=5e-4, ALIs_over_training=1)

    agent = Agent(gamma=0.99, epsilon=0.8, batch_size=128, n_actions=n_actions, eps_end=0.01,
              input_dims=[observation_shape], lr=0.01, eps_dec=5e-6*2.0, ALIs_over_training=2, file_label="DDQN_Discrete") # changed from eps_dec=5e-4
    agent.load_models()

    env = gym.make(ENV_NAME).unwrapped
    env.set_agent(0)
    action = [0.5, 0] # linear_velocity, angular_velocity. from 0 to 1, a % of the max_linear_vel (0.8) & max_angular_vel (1.8)
    start = time.time()
    scores, eps_history = [], []
    n_games = 10000
    best_score = 0
    for i in range(n_games):
        # env.person.pause() # weird side effect for ending episode (path finished)
        # env.person.resume()
        score = 0
        observation = env.reset()
        done = False
        while not done:
            rel_heading = env.get_relative_heading_position(env.robot, env.person)[1]
            orientation_rad = np.arctan2(rel_heading[1], rel_heading[0])
            action_set = compute_action_set(orientation_rad)

            action_idx = agent.choose_action(observation)
            action = action_set[action_idx] # un-discretized

            observation_, reward, done, _ = env.step(action)
            # dx_dt, dy_dt, da_dt = env.get_system_velocities() # best to see code. (dx_dt, dy_dt, da_dt)
            # print(f'X: {dx_dt} | Y: {dy_dt} | Angular V: {da_dt}')

            # Prints out x y heading position of person
            # person_state = env.get_person_pos()  # [xy[0], xy[1], theta] where theta is orientation
            # print(f'Person state is {person_state}')

            # print(f'State is {state}') # shape is 47

            # print(f"Robot state \n\t position is {env.robot.state_['position']} \n\t orientation is {env.robot.state_['orientation']} \n\t velocity lin & angular is {env.robot.state_['velocity']}")
            # print(f'Person state\n\t position is {env.person.state_["position"]}\n\t orientation is {env.person.state_["orientation"]}\n\t velocity lin & angular is {env.person.state_["velocity"]}')

            # rel_pos = env.get_relative_position(env.person.get_pos(), env.robot)
            # distance = np.hypot(rel_pos[0], rel_pos[1])

            # print(f'get relative position. person.pos()-robot.pos(): {rel_pos} | with a distance of distance')

            score += reward

            agent.store_transition(observation, action_idx, reward, observation_, done)
            agent.learn()
            observation = observation_
            # while episode

        # if score > best_score:
        #     best_score = score
        #     agent.save_models()

        if i % 50 == 0:
            agent.save_models() # TODO necessary evil
        
        # if i % 1 == 0:
        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-10:])
        print('episode ', i, 'score %.2f' % score, 'average score %.2f' % avg_score, 'epsilon %.2f' % agent.epsilon)
        #end of game  

    end = time.time()
    print(f'Time taken: {(end - start):.4f}')    
    print("END")


