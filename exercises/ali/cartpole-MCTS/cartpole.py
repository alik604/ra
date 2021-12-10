# from https://github.com/kvwoerden/mcts-cartpole
# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import os
import time
import random
import argparse
from types import SimpleNamespace

import gym
from gym import logger
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from Simple_mcts import MCTSAgent
from Agent import dqn_agent
# ---------------------------------------------------------------------------- #
#                                   Constants                                  #
# ---------------------------------------------------------------------------- #
LOGGER_LEVEL = logger.WARN

args = dict()
args['env_name'] = 'CartPole-v0'
args['episodes'] = 10
args['seed'] = 28
args['iteration_budget'] = 8000       # The number of iterations for each search step. Increasing this should lead to better performance.')
args['lookahead_target'] = 10000     # The target number of steps the agent aims to look forward.'
args['max_episode_steps'] = 1500    # The maximum number of steps to play.
args['video_basepath'] = '.\\video' # './video'
args['start_cp'] = 20  # The start value of C_p, the value that the agent changes to try to achieve the lookahead target. Decreasing this makes the search tree deeper, increasing this makes the search tree wider.
args = SimpleNamespace(**args)
# ---------------------------------------------------------------------------- #
#                                   Main loop                                  #
# ---------------------------------------------------------------------------- #
if __name__ == '__main__':

    logger.set_level(LOGGER_LEVEL)
    random.seed(args.seed)

    env = gym.make(args.env_name)
    env.seed(args.seed)

    Q_net = dqn_agent()
    agent = MCTSAgent(args.iteration_budget, env, Q_net)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    reward = 0
    done = False

    for i in range(args.episodes):
        ob = env.reset()
        env._max_episode_steps = args.max_episode_steps
        video_path = os.path.join(
            args.video_basepath, f"output_{timestr}_{i}.mp4")
        # rec = VideoRecorder(env, path=video_path)

        try:
            sum_reward = 0
            node = None
            all_nodes = []
            C_p = args.start_cp
            while True:
                print("################")
                env.render()
                # rec.capture_frame()
                action, node, C_p = agent.act(env.state, n_actions=env.action_space.n, node=node, C_p=C_p, lookahead_target=args.lookahead_target)
                ob, reward, done, _ = env.step(action)
                print("### observed state: ", ob)
                sum_reward += reward
                print("### sum_reward: ", sum_reward)
                if done:
                    # rec.close()
                    break

        except KeyboardInterrupt as e:
            # rec.close()
            env.close()
            raise e

    env.close()
