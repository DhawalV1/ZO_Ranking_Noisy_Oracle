import gymnasium as gym
import numpy as np


class MujocoEvaluator:

    def __init__(
        self,
        env_name="Reacher-v4",
        max_episode_steps=None
    ):

        self.env_name = env_name

        self.max_episode_steps = max_episode_steps

    def evaluate_policy(
        self,
        policy,
        num_episodes=1
    ):

        rewards = []

        for _ in range(num_episodes):

            env = gym.make(
                self.env_name,
                render_mode = None
            )

            obs, _ = env.reset()

            done = False
            truncated = False

            total_reward = 0.0
            step_count = 0

            while not (done or truncated):

                action = policy.act(obs)

                obs, reward, done, truncated, _ = env.step(
                    action
                )

                total_reward += reward

                step_count += 1

                if (
                    self.max_episode_steps
                    is not None
                    and
                    step_count >= self.max_episode_steps
                ):
                    break

            env.close()

            rewards.append(
                total_reward
            )

        return np.mean(rewards)