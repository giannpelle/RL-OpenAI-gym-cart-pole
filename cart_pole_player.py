import os, sys
sys.path.insert(1, os.path.abspath("agents/"))

from agents.DoubleDQN_agent import DoubleDQNAgent
from agents.PerDoubleDQN_agent import PerDoubleDQNAgent
from agents.DuelingPerDoubleDQN_agent import DuelingPerDoubleDQNAgent

import argparse
import gym

parser = argparse.ArgumentParser()
parser.add_argument('--method', 
                    default='doubledqn',
                    help='specify the RL technique to be used')
arguments = parser.parse_args()

maze_env = gym.make("CartPole-v0")

def run_double_dqn_method(*, env):
    double_dqn_agent = DoubleDQNAgent(env=env)
    rewards = double_dqn_agent.off_policy_train()
    double_dqn_agent.run_optimal()

def run_per_double_dqn_method(*, env):
    per_double_dqn_agent = PerDoubleDQNAgent(env=env)
    rewards = per_double_dqn_agent.off_policy_train()
    per_double_dqn_agent.run_optimal()

def run_dueling_per_double_dqn_method(*, env):
    dueling_per_double_dqn_agent = DuelingPerDoubleDQNAgent(env=env)
    rewards = dueling_per_double_dqn_agent.off_policy_train()
    dueling_per_double_dqn_agent.run_optimal()

switcher = {
    "doubledqn": run_double_dqn_method,
    "perdoubledqn": run_per_double_dqn_method,
    "duelingperdoubledqn": run_dueling_per_double_dqn_method
}
solver = switcher.get(arguments.method, run_double_dqn_method)
solver(env=maze_env)







