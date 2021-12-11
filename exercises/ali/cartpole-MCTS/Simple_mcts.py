# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import random
import math
import itertools
import random
import numpy as np
<<<<<<< HEAD

import gym


=======
import torch
import gym

>>>>>>> MCTS
# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #
DEBUG = False


# ---------------------------------------------------------------------------- #
#                            Monte Carlo Tree Search                           #
# ---------------------------------------------------------------------------- #
class MCTSNode:
    id_iter = itertools.count()

<<<<<<< HEAD
    def __init__(self, params, done, depth):
        self.params = params
        self.children = {}
        self.parent = None
        self.Q = 0 # Number of wins for the node considered after the i-th move
        self.N = 0 # Number of simulations for the node considered after the i-th move
=======
    def __init__(self, state, done, depth, prior=1):
        self.state = state
        self.children = {}
        self.parent = None
        self.Q = 0  # Number of wins for the node considered after the i-th move
        self.N = 0  # Number of simulations for the node considered after the i-th move
>>>>>>> MCTS
        self.id = next(MCTSNode.id_iter)
        self.done = done
        self.depth = depth

<<<<<<< HEAD
=======
        self.prior = prior

>>>>>>> MCTS
    def add_child(self, action, child):
        self.children[action] = child


<<<<<<< HEAD
def UctSearch(params, n_actions, environment, iterations=100, node=None,  C_p=None, lookahead_target=None):
    if C_p == None:
        C_p = 200
    if lookahead_target == None:
        lookahead_target = 200
    if node == None:
        root_node = MCTSNode(params, False, 0)
=======
def UctSearch(state, n_actions, env, DQNAgent, iterations=10, node=None, C_p=None, lookahead_target=None):
    if C_p is None:
        C_p = 200
    if lookahead_target is None:
        lookahead_target = 200
    if node is None:
        root_node = MCTSNode(state, False, 0)
>>>>>>> MCTS
    else:
        root_node = node

    counter = 0
    max_depth = 0
    ix = 0
    while True:
<<<<<<< HEAD
        v = TreePolicy(root_node, C_p, n_actions, environment)
        max_depth = max(v.depth - root_node.depth, max_depth)
        Delta = DefaultPolicy(v, environment) # reward from single rollout of random actions from State S (
=======
        v = TreePolicy(root_node, C_p, n_actions, env, DQNAgent)
        max_depth = max(v.depth - root_node.depth, max_depth)
        # Delta = DefaultPolicy(v, n_actions, env)  # reward from single rollout of random actions from State S
        Delta = DQNAgentPolicy(v, DQNAgent)  # instead of needing env, we use the DQNAgent.net.state_value_output, which is the expected value of the state
        # print(f'Delta is {Delta}')
>>>>>>> MCTS
        Backup(v, Delta, root_node)
        counter += 1
        ix += 1
        if ix > iterations:
            break
    if max_depth < lookahead_target:
        C_p = C_p - 1
    else:
        C_p = C_p + 1
<<<<<<< HEAD
=======

>>>>>>> MCTS
    print(f"### max_depth: {max_depth:03}, lookahead_target: {lookahead_target:03} ")
    print(f"### C_p: {C_p} ")
    print("### Maximal depth considered: ", max_depth)
    for action, child in sorted(root_node.children.items()):
<<<<<<< HEAD
        print(f"### action: {action}, Q: {int(child.Q):08}, N: {child.N:08}, Q/N: {child.Q/child.N:07.2f}")

    best_child = max(root_node.children.values(), key=lambda x: x.N)
    best_child_action = best_child.action
    print(f"### predicted state: {best_child.params}")
=======
        print(f"### action: {action}, Q: {int(child.Q):08}, N: {child.N:08}, Q/N: {child.Q / child.N:07.2f}")

    best_child = max(root_node.children.values(), key=lambda x: x.N)
    best_child_action = best_child.action
    print(f"### predicted state: {best_child.state}")
>>>>>>> MCTS
    print(f"### chosen action: {best_child_action}")

    best_child_node = max(root_node.children.values(), key=lambda x: x.N)
    return (best_child_node.action, best_child_node, C_p)


<<<<<<< HEAD
def TreePolicy(node, C_p, n_actions, environment):
    while not node.done:
        if len(node.children) < n_actions:
            return Expand(node, n_actions, environment)
=======
def TreePolicy(node, C_p, n_actions, env, DQNAgent):
    while not node.done:
        if len(node.children) < n_actions:
            # print(f'Expanding...  {len(node.children)}')
            return Expand(node, n_actions, env, DQNAgent)
>>>>>>> MCTS
        else:
            node = BestChild(node, C_p)
    return node


<<<<<<< HEAD
def Expand(node, n_actions, environment):
    exp_env = gym.make(environment)
    exp_env.reset()
    exp_env.unwrapped.state = np.array(node.params) # I think this means set a global state to X

    unchosen_actions = list(filter(lambda action: not action in node.children.keys(), range(n_actions))) # every action not expanded yet (..in node)
    a = random.choice(unchosen_actions)
    params, _, done, _ = exp_env.step(a)
    child_node = MCTSNode(params, done, node.depth+1)
    child_node.parent = node
    node.children[a] = child_node
    child_node.action = a
=======
def Expand(node, n_actions, env, DQNAgent):
    # exp_env = gym.make(environment)
    # exp_env.reset()
    # exp_env.unwrapped.state = np.array(node.state)  # I think this means set a global state to X
    # unchosen_actions = list(filter(lambda action: not action in node.children.keys(), range(n_actions)))  # every action not expanded yet (..in node)
    # a = random.choice(unchosen_actions)

    with torch.no_grad():
        data = torch.tensor(node.state, dtype=torch.float32)
        action_values = DQNAgent.net(data)

    action, prob_action = DQNAgent.select_actions_boltzmann(action_values, n_actions=range(n_actions), temperature=5, return_action_and_prob=True) # weighted sampled action, per probablities p
    state_, _, done, _ = env.step(action)
    child_node = MCTSNode(state_, done, node.depth + 1)
    child_node.parent = node
    child_node.action = action
    child_node.prior = prob_action # probablity of the node's action

    node.children[action] = child_node
>>>>>>> MCTS
    return child_node


def BestChild(node, c, random=False):
    if random:
<<<<<<< HEAD
        child_values = {child: child.Q/child.N + c *
                        math.sqrt(2*math.log(node.N)/child.N) for child in node.children}
=======
        child_values = {child: child.Q / child.N + c *
                               math.sqrt(2 * math.log(node.N) / child.N) for child in node.children}
>>>>>>> MCTS
        mv = max(child_values.values())
        am = random.choice([k for (k, v) in child_values.items() if v == mv])
    else:
        am = max(node.children.values(), key=lambda v_prime: v_prime.Q /
<<<<<<< HEAD
                 v_prime.N + c*math.sqrt(2*math.log(node.N)/v_prime.N))
    return am


def DefaultPolicy(node, environment):
    exp_env = gym.make(environment)
    exp_env.reset()
    exp_env.unwrapped.state = np.array(node.params)
=======
                                                             v_prime.N + c * math.sqrt(2 * math.log(node.N) / v_prime.N))
    return am




def DQNAgentPolicy(node, DQNAgent):
    done = node.done
    reward = node.depth
    if done:
        return reward

    # return DQNAgent.net.state_value_output
    with torch.no_grad():
        data = torch.tensor(node.state, dtype=torch.float32)
        action_value = DQNAgent.net(data)
        return reward + DQNAgent.net.state_value_output.numpy()[0]

def DefaultPolicy(node, n_actions, env):
>>>>>>> MCTS

    done = node.done
    reward = node.depth

    while not done:
<<<<<<< HEAD
        random_action = random.choice([0, 1])
        _, step_reward, done, _ = exp_env.step(random_action)
=======
        random_action = random.choice(range(n_actions))
        _, step_reward, done, _ = env.step(random_action)
>>>>>>> MCTS
        reward += step_reward

    return reward


def Backup(node, Delta, root_node):
    while not node is root_node.parent:
        node.N += 1
        node.Q = node.Q + Delta
        node = node.parent


class MCTSAgent(object):
<<<<<<< HEAD
    def __init__(self, time_budget, environment):
        self.agenttype = 'mcts'
        self.time_budget = time_budget
        self.environment = environment

    def act(self, params, n_actions, node, C_p, lookahead_target):
        return UctSearch(params, n_actions, self.environment, self.time_budget, node=node, C_p=C_p, lookahead_target=lookahead_target)
=======
    def __init__(self, time_budget, environment, DQNAgent):
        self.time_budget = time_budget
        self.environment = environment
        self.DQNAgent = DQNAgent
    def act(self, params, n_actions, node, C_p, lookahead_target):
        return UctSearch(params, n_actions, self.environment, self.DQNAgent, self.time_budget, node=node, C_p=C_p, lookahead_target=lookahead_target)
>>>>>>> MCTS
