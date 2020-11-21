import gym
import numpy as np

from PPO.replay import cumulative_sum, Episode, History


def test_cumulative_sum_1():
    array = [0, 1, 2, 3, 4, 5]

    cumulative_array = cumulative_sum(array)

    expected_cumulative_array = [15, 15, 14, 12, 9, 5]

    assert cumulative_array == expected_cumulative_array


def test_cumulative_sum_2():
    array = [0, 1, 2, 3, 4, 5]

    cumulative_array = cumulative_sum(array, gamma=0.99)

    expected_cumulative_array = [
        14.458431289499998,
        14.604476049999999,
        13.741895,
        11.8605,
        8.95,
        5.0,
    ]

    assert cumulative_array == expected_cumulative_array


def test_episode_1():
    episode = Episode(gamma=0.99, lambd=0.95)

    reward_scale = 20

    episode.append(
        observation=0,
        action=1,
        reward=0,
        value=0,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.append(
        observation=0,
        action=1,
        reward=1,
        value=0,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.append(
        observation=0,
        action=1,
        reward=2,
        value=0,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.append(
        observation=0,
        action=1,
        reward=3,
        value=0,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.append(
        observation=0,
        action=1,
        reward=4,
        value=0,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.append(
        observation=0,
        action=1,
        reward=5,
        value=0,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.end_episode(last_value=0)

    expected_rewards_to_go = [
        0.722921564475,
        0.7302238025,
        0.68709475,
        0.593025,
        0.4475,
        0.25,
    ]

    assert episode.rewards_to_go == expected_rewards_to_go


def test_episode_2():
    episode = Episode(gamma=0.99, lambd=0.95)

    reward_scale = 20

    episode.append(
        observation=0,
        action=1,
        reward=0,
        value=0,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.append(
        observation=0,
        action=1,
        reward=1,
        value=0,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.append(
        observation=0,
        action=1,
        reward=2,
        value=1,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.append(
        observation=0,
        action=1,
        reward=3,
        value=2,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.append(
        observation=0,
        action=1,
        reward=4,
        value=3,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.append(
        observation=0,
        action=1,
        reward=5,
        value=5,
        log_probability=-1,
        reward_scale=reward_scale,
    )
    episode.end_episode(last_value=5)

    expected_advantages = [
        4.694519008033593,
        4.991514096792763,
        4.201503558525,
        3.3189830500000004,
        2.3381000000000007,
        0.20000000000000018,
    ]
    assert episode.advantages == expected_advantages


def test_history_1():
    episode1 = Episode(gamma=0.99, lambd=0.95)
    episode1.append(observation=0, action=1, reward=0, value=0, log_probability=-1)
    episode1.append(observation=0, action=1, reward=1, value=0, log_probability=-1)
    episode1.append(observation=0, action=1, reward=2, value=1, log_probability=-1)
    episode1.append(observation=0, action=1, reward=3, value=2, log_probability=-1)
    episode1.append(observation=0, action=1, reward=4, value=3, log_probability=-1)
    episode1.append(observation=0, action=1, reward=5, value=5, log_probability=-1)
    episode1.end_episode(last_value=5)

    episode2 = Episode(gamma=0.99, lambd=0.95)
    episode2.append(observation=0, action=1, reward=0, value=0, log_probability=-1)
    episode2.append(observation=0, action=1, reward=-1, value=0, log_probability=-1)
    episode2.append(observation=0, action=1, reward=-2, value=-1, log_probability=-1)
    episode2.append(observation=0, action=1, reward=3, value=2, log_probability=-1)
    episode2.append(observation=0, action=1, reward=-4, value=-3, log_probability=-1)
    episode2.end_episode(last_value=0)

    history = History()

    history.add_episode(episode1)
    history.add_episode(episode2)

    history.build_dataset()

    assert len(history) == 11
    assert abs(np.mean(history.advantages)) <= 1e-10
    assert abs(np.std(history.advantages) - 1) <= 1e-3


def test_history_2():
    episode1 = Episode(gamma=0.99, lambd=0.95)
    episode1.append(observation=0, action=1, reward=0, value=0, log_probability=-1)
    episode1.append(observation=0, action=1, reward=1, value=0, log_probability=-1)
    episode1.append(observation=0, action=1, reward=2, value=1, log_probability=-1)
    episode1.append(observation=0, action=1, reward=3, value=2, log_probability=-1)
    episode1.append(observation=0, action=1, reward=4, value=3, log_probability=-1)
    episode1.append(observation=0, action=1, reward=5, value=5, log_probability=-1)
    episode1.end_episode(last_value=5)

    episode2 = Episode(gamma=0.99, lambd=0.95)
    episode2.append(observation=0, action=1, reward=0, value=0, log_probability=-1)
    episode2.append(observation=0, action=1, reward=-1, value=0, log_probability=-1)
    episode2.append(observation=0, action=1, reward=-2, value=-1, log_probability=-1)
    episode2.append(observation=0, action=1, reward=3, value=2, log_probability=-1)
    episode2.append(observation=0, action=1, reward=-4, value=-3, log_probability=-1)
    episode2.end_episode(last_value=0)

    history = History()

    history.add_episode(episode1)
    history.add_episode(episode2)

    history.build_dataset()

    history.free_memory()

    assert len(history) == 0
    assert len(history.rewards) == 0
    assert len(history.advantages) == 0
    assert len(history.log_probabilities) == 0
    assert len(history.rewards_to_go) == 0
    assert len(history.episodes) == 0


def test_history_episode():
    reward_scale = 20

    env = gym.make("LunarLander-v2")
    observation = env.reset()

    n_actions = env.action_space.n
    feature_dim = observation.size

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

    assert np.std(history.log_probabilities) <= 1e-3

    assert (
        abs(
            sum([v for episode in history.episodes for v in episode.values])
            - ite * (ite - 1) / 2
        )
        <= 1e-3
    )

    assert history.observations[-1].shape[0] == feature_dim

    assert (
        abs(
            len([a for a in history.actions if a == 0])
            - len(history.actions) / n_actions
        )
        < 30
    )
