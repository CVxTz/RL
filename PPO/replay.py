import numpy as np

from torch.utils.data import Dataset


def cumulative_sum(array, gamma=1.0):
    curr = 0
    cumulative_array = []

    for a in array[::-1]:
        curr = a + gamma * curr
        cumulative_array.append(curr)

    return cumulative_array[::-1]


class Episode:
    def __init__(self, gamma=0.99, lambd=0.95):
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.values = []
        self.log_probabilities = []
        self.gamma = gamma
        self.lambd = lambd

    def append(
        self, observation, action, reward, value, log_probability, reward_scale=20
    ):
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward / reward_scale)
        self.values.append(value)
        self.log_probabilities.append(log_probability)

    def end_episode(self, last_value):
        rewards = np.array(self.rewards + [last_value])
        values = np.array(self.values + [last_value])

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantages = cumulative_sum(deltas.tolist(), gamma=self.gamma * self.lambd)

        self.rewards_to_go = cumulative_sum(rewards.tolist(), gamma=self.gamma)[:-1]


def normalize_list(array):
    array = np.array(array)
    array = (array - np.mean(array)) / (np.std(array) + 1e-5)
    return array.tolist()


class History(Dataset):
    def __init__(self):
        self.episodes = []
        self.observations = []
        self.actions = []
        self.advantages = []
        self.rewards = []
        self.rewards_to_go = []
        self.log_probabilities = []

    def free_memory(self):
        del self.episodes[:]
        del self.observations[:]
        del self.actions[:]
        del self.advantages[:]
        del self.rewards[:]
        del self.rewards_to_go[:]
        del self.log_probabilities[:]

    def add_episode(self, episode):
        self.episodes.append(episode)

    def build_dataset(self):
        for episode in self.episodes:
            self.observations += episode.observations
            self.actions += episode.actions
            self.advantages += episode.advantages
            self.rewards += episode.rewards
            self.rewards_to_go += episode.rewards_to_go
            self.log_probabilities += episode.log_probabilities

        assert (
            len(
                {
                    len(self.observations),
                    len(self.actions),
                    len(self.advantages),
                    len(self.rewards),
                    len(self.rewards_to_go),
                    len(self.log_probabilities),
                }
            )
            == 1
        )

        self.advantages = normalize_list(self.advantages)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return (
            self.observations[idx],
            self.actions[idx],
            self.advantages[idx],
            self.log_probabilities[idx],
            self.rewards_to_go[idx],
        )
