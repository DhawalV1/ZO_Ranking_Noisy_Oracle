import gymnasium as gym
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from policies.mlp_policy import MLPPolicy

from envs.mujoco_env import MujocoEvaluator

# from oracles.perfect_oracle import PerfectOracle
from oracles.perfect_oracle import PerfectOracle
from rankzo.graph_builder import build_edges
from rankzo.estimator import estimate_gradient
from rankzo.optimizer import ZORankSGD
import random
import torch

RESULT_DIR = "results_MLP_wo_noise"

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

SEED = 0

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

SEEDS = [0,1,2]
all_reward_histories = []

all_initial_rewards = []
all_best_rewards = []
all_final_rewards = []

all_std_histories = []
ENV_NAME = "Reacher-v4"

T = 250

m = 100

mu = 0.001

lr = 0.01


###################################################
# Setup
###################################################

env = gym.make(ENV_NAME)

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

env.close()

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
for s in SEEDS:
    oracle.reset_stats()
    random.seed(s)
    torch.manual_seed(s)

    policy = MLPPolicy(
        obs_dim,
        act_dim
    )

    theta = policy.get_parameters()

    d = len(theta)
    print(d)

    reward_history = []

    std_history = []

    best_reward = -np.inf
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
                num_episodes=3
            )

            returns.append(
                reward
            )
        
        if t % 10 == 0:

            current_std = np.std(returns)

            std_history.append(
                current_std
            )

            print(
                f"Return std: {current_std:.4f}"
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
            reward_history.append(
                eval_reward
            )

            best_reward = max(
                best_reward,
                eval_reward
            )
            print(
                f"Iter {t:4d} | "
                f"Reward {eval_reward:.2f}"
            )

    all_reward_histories.append(
        reward_history
    )

    all_std_histories.append(
        std_history
    )

    all_initial_rewards.append(
        reward_history[0]
    )

    all_best_rewards.append(
        best_reward
    )

    all_final_rewards.append(
        reward_history[-1]
    )

with open(
    f"{RESULT_DIR}/summary.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(
        [
            "seed",
            "initial_reward",
            "best_reward",
            "final_reward"
        ]
    )

    for idx, seed in enumerate(SEEDS):

        writer.writerow(
            [
                seed,
                all_initial_rewards[idx],
                all_best_rewards[idx],
                all_final_rewards[idx]
            ]
        )
reward_matrix = np.array(
    all_reward_histories
)

mean_reward = reward_matrix.mean(
    axis=0
)

std_reward = reward_matrix.std(
    axis=0
)

x = np.arange(
    len(mean_reward)
) * 10
plt.figure(figsize=(8,5))

plt.plot(
    x,
    mean_reward,
    label="Mean Reward"
)

plt.fill_between(
    x,
    mean_reward - std_reward,
    mean_reward + std_reward,
    alpha=0.3
)

plt.xlabel("Iteration")
plt.ylabel("Reward")

plt.title(
    "ZO-RankSGD Learning Curve"
)

plt.grid(True)

plt.legend()

plt.savefig(
    f"{RESULT_DIR}/reward_curve.png"
)

plt.close()

std_matrix = np.array(
    all_std_histories
)

mean_std = std_matrix.mean(
    axis=0
)

plt.figure(figsize=(8,5))

plt.plot(
    x,
    mean_std
)

plt.xlabel("Iteration")
plt.ylabel("Return Std")

plt.title(
    "Candidate Return Spread"
)

plt.grid(True)

plt.savefig(
    f"{RESULT_DIR}/return_std_curve.png"
)

plt.close()

with open(
    f"{RESULT_DIR}/reward_history.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    header = ["iteration"]

    for seed in SEEDS:
        header.append(
            f"seed_{seed}"
        )

    writer.writerow(header)

    reward_matrix = np.array(
        all_reward_histories
    )

    for i in range(
        reward_matrix.shape[1]
    ):

        row = [i*10]

        row.extend(
            reward_matrix[:, i]
        )

        writer.writerow(row)

plt.figure(figsize=(7,5))

plt.hist(
    all_best_rewards,
    bins=10
)

plt.xlabel("Best Reward")
plt.ylabel("Frequency")

plt.title(
    "Distribution of Best Rewards"
)

plt.savefig(
    f"{RESULT_DIR}/best_reward_hist.png"
)

plt.close()