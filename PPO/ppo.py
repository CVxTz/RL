from pathlib import Path

import gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from PPO.model import (
    PolicyNetwork,
    ValueNetwork,
    device,
    train_value_network,
    train_policy_network,
)
from PPO.replay import Episode, History


def main(
    env_name="LunarLander-v2",
    reward_scale=20.0,
    clip=0.2,
    log_dir="../logs",
    learning_rate=0.001,
    state_scale=1.0,
):
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=env_name, comment=env_name)

    env = gym.make(env_name)
    observation = env.reset()

    n_actions = env.action_space.n
    feature_dim = observation.size

    value_model = ValueNetwork(in_dim=feature_dim).to(device)
    value_optimizer = optim.Adam(value_model.parameters(), lr=learning_rate)

    policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions).to(device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=learning_rate)

    n_epoch = 4

    max_episodes = 20
    max_timesteps = 400

    batch_size = 32

    max_iterations = 200

    history = History()

    epoch_ite = 0
    episode_ite = 0

    for ite in tqdm(range(max_iterations)):

        if ite % 50 == 0:
            torch.save(
                policy_model.state_dict(),
                Path(log_dir) / (env_name + f"_{str(ite)}_policy.pth"),
            )
            torch.save(
                value_model.state_dict(),
                Path(log_dir) / (env_name + f"_{str(ite)}_value.pth"),
            )

        for episode_i in range(max_episodes):

            observation = env.reset()
            episode = Episode()

            for timestep in range(max_timesteps):

                action, log_probability = policy_model.sample_action(
                    observation / state_scale
                )
                value = value_model.state_value(observation / state_scale)

                new_observation, reward, done, info = env.step(action)

                episode.append(
                    observation=observation / state_scale,
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
                    value = value_model.state_value(observation / state_scale)
                    episode.end_episode(last_value=value)

            episode_ite += 1
            writer.add_scalar(
                "Average Episode Reward",
                reward_scale * np.sum(episode.rewards),
                episode_ite,
            )
            writer.add_scalar(
                "Average Probabilities",
                np.exp(np.mean(episode.log_probabilities)),
                episode_ite,
            )

            history.add_episode(episode)

        history.build_dataset()
        data_loader = DataLoader(history, batch_size=batch_size, shuffle=True)

        policy_loss = train_policy_network(
            policy_model, policy_optimizer, data_loader, epochs=n_epoch, clip=clip
        )

        value_loss = train_value_network(
            value_model, value_optimizer, data_loader, epochs=n_epoch
        )

        for p_l, v_l in zip(policy_loss, value_loss):
            epoch_ite += 1
            writer.add_scalar("Policy Loss", p_l, epoch_ite)
            writer.add_scalar("Value Loss", v_l, epoch_ite)

        history.free_memory()


if __name__ == "__main__":

    main(
        reward_scale=20.0,
        clip=0.2,
        env_name="LunarLander-v2",
        learning_rate=0.001,
        state_scale=1.0,
        log_dir="logs/Lunar"
    )
