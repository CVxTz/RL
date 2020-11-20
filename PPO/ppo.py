import sys
from pathlib import Path
import numpy as np
import gym
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from PPO.model import (
    PolicyNetwork,
    ValueNetwork,
    device,
    train_value_network,
    train_policy_network,
)
from PPO.replay import Episode, History


def main():

    env = gym.make("LunarLander-v2")
    observation = env.reset()

    n_actions = env.action_space.n
    feature_dim = observation.size

    value_model = ValueNetwork(in_dim=feature_dim).to(device)
    value_optimizer = optim.Adam(value_model.parameters(), lr=0.001)

    policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions).to(device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=0.001)

    n_epoch = 4

    max_episodes = 20
    max_timesteps = 300

    max_iterations = 10000

    reward_scale = 20

    history = History()

    for iteration in range(max_iterations):

        for episode_i in range(max_episodes):

            observation = env.reset()
            episode = Episode()

            for timestep in range(max_timesteps):

                action, log_probability = policy_model.sample_action(observation)
                value = value_model.state_value(observation)

                new_observation, reward, done, info = env.step(action)

                episode.append(
                    observation=observation,
                    action=action,
                    reward=reward,
                    value=value,
                    log_probability=log_probability,
                    reward_scale=reward_scale,
                )

                observation = new_observation

                if done:
                    episode.end_episode(last_value=0)
                    break

                if timestep == max_timesteps - 1:
                    value = value_model.state_value(observation)
                    episode.end_episode(last_value=value)

            history.add_episode(episode)

        history.build_dataset()
        data_loader = DataLoader(history, batch_size=64, shuffle=True)

        train_policy_network(
            policy_model, policy_optimizer, data_loader, epochs=n_epoch
        )

        train_value_network(value_model, value_optimizer, data_loader, epochs=n_epoch)

        print(iteration, "Mean reward", reward_scale * np.sum(history.rewards) / max_episodes)

        history.free_memory()


if __name__ == "__main__":

    main()
