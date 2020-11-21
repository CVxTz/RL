import gym
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from PPO.model import (
    PolicyNetwork,
    ValueNetwork,
    device,
    train_value_network,
    train_policy_network,
)
from PPO.replay import Episode, History


def test_model_1():
    env = gym.make("LunarLander-v2")

    observation = env.reset()

    n_actions = env.action_space.n
    feature_dim = observation.size

    policy_model = PolicyNetwork(n=n_actions, in_dim=feature_dim)

    policy_model.to(device)

    action, log_probability = policy_model.sample_action(observation)

    assert action in list(range(n_actions))


def test_model_2():
    env = gym.make("LunarLander-v2")

    observation = env.reset()

    n_actions = env.action_space.n
    feature_dim = observation.size

    policy_model = PolicyNetwork(n=n_actions, in_dim=feature_dim)
    policy_model.to(device)

    observations = [observation / i for i in range(1, 11)]

    observations = torch.from_numpy(np.array(observations)).to(device)

    probs = policy_model(observations)

    assert list(probs.size()) == [10, n_actions]

    assert abs(probs[0, :].sum().item() - 1) < 1e-3


def test_model_3():
    env = gym.make("LunarLander-v2")

    observation = env.reset()

    n_actions = env.action_space.n
    feature_dim = observation.size

    policy_model = PolicyNetwork(n=n_actions, in_dim=feature_dim)
    policy_model.to(device)

    observations = [observation / i for i in range(1, 11)]

    actions = [i % 4 for i in range(1, 11)]

    observations = torch.from_numpy(np.array(observations)).to(device)
    actions = torch.IntTensor(actions).to(device)

    log_probabilities, entropy = policy_model.evaluate_actions(observations, actions)

    assert list(log_probabilities.size()) == [10]
    assert list(entropy.size()) == [10]


def test_history_episode_model():
    reward_scale = 20

    env = gym.make("LunarLander-v2")
    observation = env.reset()

    n_actions = env.action_space.n
    feature_dim = observation.size

    policy_model = PolicyNetwork(n=n_actions, in_dim=feature_dim).to(device)
    value_model = ValueNetwork(in_dim=feature_dim).to(device)

    max_episodes = 10
    max_timesteps = 100

    reward_sum = 0
    ite = 0

    history = History()

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

            reward_sum += reward
            ite += 1

            if done:
                episode.end_episode(last_value=np.random.uniform())
                break

            if timestep == max_timesteps - 1:
                episode.end_episode(last_value=0)

        history.add_episode(episode)

    history.build_dataset()

    assert abs(np.sum(history.rewards) - reward_sum / reward_scale) < 1e-5

    assert len(history.rewards) == ite

    assert abs(np.mean(history.advantages)) <= 1e-10

    assert abs(np.std(history.advantages) - 1) <= 1e-3


def test_value_network():
    env = gym.make("LunarLander-v2")
    observation = env.reset()

    n_actions = env.action_space.n
    feature_dim = observation.size

    n_epoch = 4

    max_episodes = 10
    max_timesteps = 100

    reward_sum = 0
    ite = 0

    history = History()

    for episode_i in range(max_episodes):

        observation = env.reset()
        episode = Episode()

        for timestep in range(max_timesteps):

            action = env.action_space.sample()

            new_observation, reward, done, info = env.step(action)

            episode.append(
                observation=observation,
                action=action,
                reward=reward,
                value=ite,
                log_probability=np.log(1 / n_actions),
            )

            observation = new_observation

            reward_sum += reward
            ite += 1

            if done:
                episode.end_episode(last_value=np.random.uniform())
                break

            if timestep == max_timesteps - 1:
                episode.end_episode(last_value=0)

        history.add_episode(episode)

    history.build_dataset()

    value_model = ValueNetwork(in_dim=feature_dim).to(device)
    value_optimizer = optim.Adam(value_model.parameters(), lr=0.001)

    data_loader = DataLoader(history, batch_size=64, shuffle=True)

    epochs_losses = train_value_network(
        value_model, value_optimizer, data_loader, epochs=n_epoch
    )

    assert epochs_losses[0] > epochs_losses[-1]


def test_policy_network():
    env = gym.make("LunarLander-v2")
    observation = env.reset()

    n_actions = env.action_space.n
    feature_dim = observation.size

    n_epoch = 4

    max_episodes = 10
    max_timesteps = 100

    reward_sum = 0
    ite = 0

    history = History()

    for episode_i in range(max_episodes):

        observation = env.reset()
        episode = Episode()

        for timestep in range(max_timesteps):

            action = env.action_space.sample()

            new_observation, reward, done, info = env.step(action)

            episode.append(
                observation=observation,
                action=action,
                reward=reward,
                value=ite,
                log_probability=np.log(1 / n_actions),
            )

            observation = new_observation

            reward_sum += reward
            ite += 1

            if done:
                episode.end_episode(last_value=np.random.uniform())
                break

            if timestep == max_timesteps - 1:
                episode.end_episode(last_value=0)

        history.add_episode(episode)

    history.build_dataset()

    policy_model = PolicyNetwork(in_dim=feature_dim).to(device)
    policy_optimizer = optim.Adam(policy_model.parameters(), lr=0.01)

    data_loader = DataLoader(history, batch_size=64, shuffle=True)

    epochs_losses = train_policy_network(
        policy_model, policy_optimizer, data_loader, epochs=n_epoch
    )

    assert epochs_losses[0] > epochs_losses[-1]
