import gymnasium as gym
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from policies.mlp_policy import MLPPolicy
from scipy.stats import kendalltau
from envs.mujoco_env import MujocoEvaluator
import argparse
# from oracles.perfect_oracle import PerfectOracle
from oracles.near_tie_oracle import NearTieOracle
from rankzo.graph_builder import build_edges
from rankzo.estimator import estimate_gradient
from rankzo.optimizer import ZORankSGD
import random
import torch

parser = argparse.ArgumentParser()

parser.add_argument(
    "--tau",
    type=float,
    required=True
)

parser.add_argument(
    "--result_dir",
    type=str,
    required=True
)

args = parser.parse_args()

TAU = args.tau
RESULT_DIR = args.result_dir

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

oracle = NearTieOracle(
    tau=TAU,
    flip_prob=0.3
)

SEEDS = [0,1,2]
all_reward_histories = []

all_initial_rewards = []
all_best_rewards = []
all_final_rewards = []

all_corruption_histories = []
all_ranking_diff_histories = []
all_std_histories = []

all_ranking_diff_histories = []

all_pair_std = []
all_pair_diff = []

ENV_NAME = "Reacher-v4"

T = 250

m = 50

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


optimizer = ZORankSGD(
    lr=lr
)


###################################################
# Training
###################################################
for s in SEEDS:
    
    np.random.seed(s)
    torch.manual_seed(s)

    policy = MLPPolicy(
        obs_dim,
        act_dim
    )

    theta = policy.get_parameters()
    oracle.reset_stats()
    d = len(theta)
    print(d)

    reward_history = []

    corruption_history = []

    std_history = []
    ranking_diff_history = []
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
        
        ranking = oracle.rank(
            returns
        )

        assert len(np.unique(ranking)) == len(ranking)

        perfect_ranking = np.argsort(
            -np.array(returns)
        )

        tau, _ = kendalltau(
            perfect_ranking,
            ranking
        )

        # print("Tau:",tau)
        # print("Returns:")
        # print(returns)

        # print("Perfect:")
        # print(perfect_ranking)

        # print("Oracle:")
        # print(ranking)

        ranking_diff_history.append(
            tau
        )
        
        if t % 10 == 0:

            current_std = np.std(returns)

            all_pair_std.append(
                current_std
            )

            all_pair_diff.append(tau)

            std_history.append(
                current_std
            )

            print(
                f"Return std: {current_std:.4f}"
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

            current_corruption = (
                oracle.corruption_rate()
            )

            corruption_history.append(
                current_corruption
            )

            print(
                "Corruption rate:",
                current_corruption
            )

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

    all_corruption_histories.append(
        corruption_history
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

    all_ranking_diff_histories.append(
        ranking_diff_history
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

corr_matrix = np.array(
    all_corruption_histories
)

mean_corr = corr_matrix.mean(
    axis=0
)

plt.figure(figsize=(8,5))

plt.plot(
    x,
    mean_corr
)

plt.xlabel("Iteration")
plt.ylabel("Corruption Rate")

plt.title(
    "Oracle Corruption Rate"
)

plt.grid(True)

plt.savefig(
    f"{RESULT_DIR}/corruption_curve.png"
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

diff_matrix = np.array(
    all_ranking_diff_histories
)

mean_diff = diff_matrix.mean(
    axis=0
)

x_diff = np.arange(
    len(mean_diff)
)

plt.figure(figsize=(8,5))

plt.plot(
    x_diff,
    mean_diff
)

plt.xlabel("Iteration")
plt.ylabel("Kendall Tau")

plt.title(
    "Ranking Disagreement"
)

plt.grid(True)

plt.savefig(
    f"{RESULT_DIR}/ranking_difference_curve.png"
)

plt.close()

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

plt.figure(figsize=(8,5))

plt.scatter(
    all_pair_std,
    all_pair_diff,
    alpha=0.5
)

plt.xlabel(
    "Return Spread"
)

plt.ylabel(
    "Kendall tau"
)

plt.title(
    "Ranking Noise vs Oracle Difficulty"
)

plt.grid(True)

plt.savefig(
    f"{RESULT_DIR}/noise_vs_spread.png"
)

plt.close()

mean_final_reward = np.mean(
    all_final_rewards
)

std_final_reward = np.std(
    all_final_rewards
)

mean_tau = np.mean(
    [np.mean(x) for x in all_ranking_diff_histories]
)

with open(
    f"{RESULT_DIR}/summary.csv",
    "w"
) as f:

    f.write(
        "tau,final_reward,final_reward_std,kendall_tau\n"
    )

    f.write(
        f"{TAU},{mean_final_reward},{std_final_reward},{mean_tau}\n"
    )
