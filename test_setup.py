# test_setup.py

import gymnasium as gym

from policies.mlp_policy import MLPPolicy
from envs.mujoco_env import MujocoEvaluator
from oracles.perfect_oracle import PerfectOracle


env = gym.make("Reacher-v4")

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

env.close()

policy = MLPPolicy(
    obs_dim,
    act_dim
)

evaluator = MujocoEvaluator(
    "Reacher-v4"
)

reward = evaluator.evaluate_policy(
    policy
)

print("Reward:", reward)

oracle = PerfectOracle()

ranking = oracle.rank(
    [-10, 3, 2, 100, 5]
)

print("Ranking:", ranking)