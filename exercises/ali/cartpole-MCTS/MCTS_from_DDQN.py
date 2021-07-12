import logging
import numpy as np
from functools import partial
import torch
from torch.nn import functional as F
from torch.optim.optimizer import Optimizer

from collections import defaultdict

# from rl_agents.agents.common.factory import safe_deepcopy_env
from Agent import dqn_agent


class MCTSAgent():
    """
        An agent that uses Monte Carlo Tree Search to plan a sequence of action in an MDP.
    """
    def __init__(self, env, config):
        config["budget"] = 400
        config["max_depth"] = 10



        self.env = None # TODO
        self.config = config
        self.prior_agent = dqn_agent
        if 'model_save' in config['prior_agent']:
            self.prior_agent.load_net(path = config['model_save'])

        prior_policy = self.agent_policy_available
        rollout_policy = self.agent_policy_available

        return MCTS(self.env, prior_policy, rollout_policy, self.config)

    def agent_policy(self, state, observation):
        # Reset prior agent environment
        self.prior_agent.env = state
        # Trigger the computation of action distribution
        self.prior_agent.act(observation)
        distribution = self.prior_agent.action_distribution(observation)
        return list(distribution.keys()), list(distribution.values())

    def agent_policy_available(self, state, observation):
        actions, probs = self.agent_policy(state, observation)
        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
            probs = np.array([probs[actions.index(a)] for a in available_actions])
            probs /= np.sum(probs)
            actions = available_actions
        return actions, probs

    def random_available_policy(state, observation):
        print(f'in random_available_policy, that should not happen')
        """
            Choose actions from a uniform distribution over currently available actions only.
        :param state: the environment state
        :param observation: the corresponding observation
        :return: a tuple containing the actions and their probabilities
        """

        if hasattr(state, 'get_available_actions'):
            available_actions = state.get_available_actions()
        else:
            available_actions = np.arange(state.action_space.n)

        probabilities = np.ones((len(available_actions))) / len(available_actions)
        return available_actions, probabilities

class MCTS():
    """
       An implementation of Monte-Carlo Tree Search, with Upper Confidence Tree exploration.
    """
    def __init__(self, env, prior_policy, rollout_policy, config=None):
        """
            New MCTS instance.

        :param config: the mcts configuration. Use default if None.
        :param prior_policy: the prior policy used when expanding and selecting nodes
        :param rollout_policy: the rollout policy used to estimate the value of a leaf node
        """
        print(f'[mcts config] {config}')
        super().__init__(config)
        self.env = env
        self.prior_policy = prior_policy
        self.rollout_policy = rollout_policy
        self.np_random = None
        self.root = None
        self.observations = []
        self.reset()
        self.seed()

        if not self.config["horizon"]:
            budget = self.config["budget"]
            gamma = self.config["gamma"]
            for episodes in range(1, int(budget)):
                if episodes * self.horizon(episodes, gamma) > budget:
                    episodes = max(episodes - 1, 1)
                    horizon = self.horizon(episodes, gamma)
                    break
            self.config["episodes"], self.config["horizon"] = episodes, horizon

    def horizon(self, episodes, gamma):
        return max(int(np.ceil(np.log(episodes) / (2 * np.log(1 / gamma)))), 1)

    def default_config(cls):
        cfg = super(MCTS, cls).default_config()
        cfg.update({
            "temperature": 2 / (1 - cfg["gamma"]),
            "closed_loop": False
        })
        return cfg

    def reset(self):
        self.root = MCTSNode(parent=None, planner=self)

    def run(self, state, observation):
        """
            Run an iteration of Monte-Carlo Tree Search, starting from a given state

        :param state: the initial environment state
        :param observation: the corresponding observation
        """
        node = self.root
        total_reward = 0
        depth = 0
        terminal = False
        state.seed(self.np_random.randint(2**30))
        while depth < self.config['horizon'] and node.children and not terminal:
            action = node.sampling_rule(temperature=self.config['temperature'])
            observation, reward, terminal, _ = self.step(state, action)
            total_reward += self.config["gamma"] ** depth * reward
            node_observation = observation if self.config["closed_loop"] else None
            node = node.get_child(action, observation=node_observation)
            depth += 1

        if not node.children \
                and depth < self.config['horizon'] \
                and (not terminal or node == self.root):
            node.expand(self.prior_policy(state, observation))

        if not terminal:
            total_reward = self.evaluate(state, observation, total_reward, depth=depth)
        node.update_branch(total_reward)

    def evaluate(self, state, observation, total_reward=0, depth=0):
        """
            Run the rollout policy to yield a sample of the value of being in a given state.

        :param state: the leaf state.
        :param observation: the corresponding observation.
        :param total_reward: the initial total reward accumulated until now
        :param depth: the initial simulation depth
        :return: the total reward of the rollout trajectory
        """
        for h in range(depth, self.config["horizon"]):
            actions, probabilities = self.rollout_policy(state, observation)
            action = self.np_random.choice(actions, 1, p=np.array(probabilities))[0]
            observation, reward, terminal, _ = self.step(state, action)
            total_reward += self.config["gamma"] ** h * reward
            if np.all(terminal):
                break
        return total_reward

    def plan(self, state, observation):
        """
            Plan an optimal sequence of actions.

            Start by updating the previously found tree with the last action performed.

        :param observation: the current state
        :return: the list of actions
        """
        for i in range(self.config['episodes']):
            if (i+1) % 10 == 0:
                print(f'{} / {}'.format(i+1, self.config['episodes']))
            self.run(safe_deepcopy_env(state), observation) # TODO
        return self.get_plan()

    def reset(self):
        self.planner.step_by_reset()
        self.remaining_horizon = 0
        self.steps = 0

    # unused - but used it a main loop
    def act(self, state):
        return self.plan(state)[0]

    def get_plan(self):
        """
            Get the optimal action sequence of the current tree by recursively selecting the best action within each
            node with no exploration.

        :return: the list of actions
        """
        actions = []
        node = self.root
        while node.children:
            action = node.selection_rule()
            actions.append(action)
            node = node.children[action]
        return actions

    # unused
    def get_visits(self):
        visits = defaultdict(int)
        for observation in self.observations:
            visits[str(observation)] += 1
        return visits

    def step(self, actions):
        """
            Handle receding horizon mechanism
        :return: whether a replanning is required
        """
        replanning_required = self.remaining_horizon == 0 or len(actions) <= 1
        if replanning_required:
            self.remaining_horizon = self.config["receding_horizon"] - 1
        else:
            self.remaining_horizon -= 1

        self.planner.step_tree(actions)
        return replanning_required

    def step_planner(self, action):
        if self.config["step_strategy"] == "prior":
            """
                Replace the MCTS tree by its subtree corresponding to the chosen action, but also convert the visit counts
                to prior probabilities and before resetting them.

            :param action: a chosen action from the root node
            """
            self.step_by_subtree(action)
            self.root.convert_visits_to_prior_in_branch()
        else:
            super().step_planner(action)

    def step_tree(self, actions):
        """
            Update the planner tree when the agent performs an action

        :param actions: a sequence of actions to follow from the root node
        """
        if self.config["step_strategy"] == "reset":
            self.step_by_reset()
        elif self.config["step_strategy"] == "subtree":
            if actions:
                self.step_by_subtree(actions[0])
            else:
                self.step_by_reset()
        else:
            self.step_by_reset()

    def step_by_reset(self):
        """
            Reset the planner tree to a root node for the new state.
        """
        self.reset()

    def step_by_subtree(self, action):
        """
            Replace the planner tree by its subtree corresponding to the chosen action.

        :param action: a chosen action from the root node
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            # The selected action was never explored, start a new tree.
            self.step_by_reset()

class MCTSNode():
    K = 1.0
    """ The value function first-order filter gain"""

    def __init__(self, parent, planner, prior=1):
        self.value = 0
        self.prior = prior
        self.parent = parent
        self.planner = planner

        self.children = {}
        """ Dict of children nodes, indexed by action labels"""

        self.count = 0
        """ Number of times the node was visited."""

    def expand(self, branching_factor):
        for a in range(branching_factor):
            self.children[a] = type(self)(self, self.planner)

    def selection_rule(self):
        if not self.children:
            return None
        # Tie best counts by best value
        actions = list(self.children.keys())
        counts = MCTSNode.all_argmax([self.children[a].count for a in actions])
        return actions[max(counts, key=(lambda i: self.children[actions[i]].get_value()))]

    def sampling_rule(self, temperature=None):
        """
            Select an action from the node.
            - if exploration is wanted with some temperature, follow the selection strategy.
            - else, select the action with maximum visit count

        :param temperature: the exploration parameter, positive or zero
        :return: the selected action
        """
        if self.children:
            actions = list(self.children.keys())
            # Randomly tie best candidates with respect to selection strategy
            indexes = [self.children[a].selection_strategy(temperature) for a in actions]
            return actions[self.random_argmax(indexes)]
        else:
            return None

    def expand(self, actions_distribution):
        """
            Expand a leaf node by creating a new child for each available action.

        :param actions_distribution: the list of available actions and their prior probabilities
        """
        actions, probabilities = actions_distribution
        for i in range(len(actions)):
            if actions[i] not in self.children:
                self.children[actions[i]] = type(self)(self, self.planner, probabilities[i])

    def update(self, total_reward):
        """
            Update the visit count and value of this node, given a sample of total reward.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.count += 1
        self.value += self.K / self.count * (total_reward - self.value)

    def update_branch(self, total_reward):
        """
            Update the whole branch from this node to the root with the total reward of the corresponding trajectory.

        :param total_reward: the total reward obtained through a trajectory passing by this node
        """
        self.update(total_reward)
        if self.parent:
            self.parent.update_branch(total_reward)

    def get_child(self, action, observation=None):
        child = self.children[action]
        if observation is not None:
            if str(observation) not in child.children:
                child.children[str(observation)] = MCTSNode(parent=child, planner=self.planner, prior=0)
            child = child.children[str(observation)]
        return child

    def selection_strategy(self, temperature):
        """
            Select an action according to its value, prior probability and visit count.

        :param temperature: the exploration parameter, positive or zero.
        :return: the selected action with maximum value and exploration bonus.
        """
        if not self.parent:
            return self.get_value()

        # return self.value + temperature * self.prior * np.sqrt(np.log(self.parent.count) / self.count)
        return self.get_value() + temperature * len(self.parent.children) * self.prior/(self.count+1)

    def convert_visits_to_prior_in_branch(self, regularization=0.5):
        """
            For any node in the subtree, convert the distribution of all children visit counts to prior
            probabilities, and reset the visit counts.

        :param regularization: in [0, 1], used to add some probability mass to all children.
                               when 0, the prior is a Boltzmann distribution of visit counts
                               when 1, the prior is a uniform distribution
        """
        self.count = 0
        total_count = sum([(child.count+1) for child in self.children.values()])
        for child in self.children.values():
            child.prior = (1 - regularization)*(child.count+1)/total_count + regularization/len(self.children)
            child.convert_visits_to_prior_in_branch()

    def get_value(self):
        return self.value

    # unused
    def breadth_first_search(root, operator=None, condition=None, condition_blocking=True):
        """
            Breadth-first search of all paths to nodes that meet a given condition

        :param root: starting node
        :param operator: will be applied to all traversed nodes
        :param condition: nodes meeting that condition will be returned
        :param condition_blocking: do not explore a node which met the condition
        :return: list of paths to nodes that met the condition
        """
        queue = [(root, [])]
        while queue:
            (node, path) = queue.pop(0)
            if (condition is None) or condition(node):
                returned = operator(node, path) if operator else (node, path)
                yield returned
            if (condition is None) or not condition_blocking or not condition(node):
                for next_key, next_node in node.children.items():
                    queue.append((next_node, path + [next_key]))

    def is_leaf(self):
        return not self.children

    def path(self):
        """
        :return: sequence of action labels from the root to the node
        """
        node = self
        path = []
        while node.parent:
            for a in node.parent.children:
                if node.parent.children[a] == node:
                    path.append(a)
                    break
            node = node.parent
        return reversed(path)

    def sequence(self):
        """
        :return: sequence of nodes from the root to the node
        """
        node = self
        path = [node]
        while node.parent:
            path.append(node.parent)
            node = node.parent
        return reversed(path)

    @staticmethod
    def all_argmax(x):
        """
        :param x: a set
        :return: the list of indexes of all maximums of x
        """
        m = np.amax(x)
        return np.nonzero(x == m)[0]

    def random_argmax(self, x):
        """
            Randomly tie-breaking arg max
        :param x: an array
        :return: a random index among the maximums
        """
        indices = MCTSNode.all_argmax(x)
        return self.planner.np_random.choice(indices)

    def __str__(self):
        return "{} (n:{}, v:{:.2f})".format(list(self.path()), self.count, self.get_value())

    def __repr__(self):
        return '<node {}>'.format(id(self))

    def get_trajectories(self, full_trajectories=True, include_leaves=True):
        """
            Get a list of visited nodes corresponding to the node subtree

        :param full_trajectories: return a list of observation sequences, else a list of observations
        :param include_leaves: include leaves or only expanded nodes
        :return: the list of trajectories
        """
        trajectories = []
        if self.children:
            for action, child in self.children.items():
                child_trajectories = child.get_trajectories(full_trajectories, include_leaves)
                if full_trajectories:
                    trajectories.extend([[self] + trajectory for trajectory in child_trajectories])
                else:
                    trajectories.extend(child_trajectories)
            if not full_trajectories:
                trajectories.append(self)
        elif include_leaves:
            trajectories = [[self]] if full_trajectories else [self]
        return trajectories

    def get_obs_visits(self, state=None):
        visits = defaultdict(int)
        updates = defaultdict(int)
        if hasattr(self, "observation"):
            for node in self.get_trajectories(full_trajectories=False,
                                              include_leaves=False):
                if hasattr(node, "observation"):
                    visits[str(node.observation)] += 1
                    if hasattr(node, "updates_count"):
                        updates[str(node.observation)] += node.updates_count
        else:  # Replay required
            for node in self.get_trajectories(full_trajectories=False,
                                              include_leaves=False):
                replay_state = safe_deepcopy_env(state)
                for action in node.path():
                    observation, _, _, _ = replay_state.step(action)
                visits[str(observation)] += 1
        return visits, updates
