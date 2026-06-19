import gymnasium as gym
import numpy as np

from policies.mlp_policy import MLPPolicy

from envs.mujoco_env import MujocoEvaluator

from oracles.perfect_oracle import PerfectOracle

from rankzo.graph_builder import build_edges
from rankzo.estimator import estimate_gradient
from rankzo.optimizer import ZORankSGD


ENV_NAME = "Reacher-v4"

T = 200

m = 5

mu = 0.02

lr = 0.05


###################################################
# Setup
###################################################

env = gym.make(ENV_NAME)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

env.close()

policy = MLPPolicy(
    obs_dim,
    act_dim
)

theta = policy.get_parameters()

d = len(theta)

print("Parameter dimension:", d)

evaluator = MujocoEvaluator(
    ENV_NAME
)

oracle = PerfectOracle()

optimizer = ZORankSGD(
    lr=lr
)


###################################################
# Training
###################################################

for t in range(T):

    noises = np.random.randn(
        m,
        d
    )

    candidate_thetas = (
        theta[None, :]
        +
        mu * noises
    )

    returns = []

    for theta_i in candidate_thetas:

        policy.set_parameters(
            theta_i
        )

        reward = evaluator.evaluate_policy(
            policy,
            num_episodes=1
        )

        returns.append(
            reward
        )

    ranking = oracle.rank(
        returns
    )

    edges = build_edges(
        ranking
    )

    g = estimate_gradient(
        noises,
        edges
    )

    theta = optimizer.step(
        theta,
        g
    )

    ################################################
    # Evaluation
    ################################################

    if t % 10 == 0:

        policy.set_parameters(
            theta
        )

        eval_reward = evaluator.evaluate_policy(
            policy,
            num_episodes=5
        )

        print(
            f"Iter {t:4d} | "
            f"Reward {eval_reward:.2f}"
        )