import random

import gym
import numpy as np
import torch
from torchvision import models


class Model(torch.nn.Module):
    def __init__(self, n=4, channels=3):
        super(Model, self).__init__()
        self.cnn = torch.nn.Conv2d(channels, 3, kernel_size=3)
        self.resnet = models.resnet18(pretrained=True)
        self.fc = torch.nn.Linear(1000, n)

    def forward(self, x):
        x = self.cnn(x)
        x = self.resnet(x)
        y = self.fc(x)

        return y


def to_gray(obs):
    print(obs.max(), obs.min())
    return obs.mean(axis=-1)


def get_state(l_obs, frames=3):
    l_obs = [to_gray(a) for a in l_obs[-frames:]]

    while len(l_obs) < frames:
        l_obs.append(l_obs[-1])

    state = np.zeros(l_obs[-1].shape + (frames,))
    for i, obs in enumerate(l_obs):
        state[..., i] = obs

    return state


def predict_action(model, state):
    state = torch.from_numpy(state).float()
    state = state.to(device)
    state = state.permute(2, 0, 1)

    state = state.unsqueeze(0)

    action = np.argmax(model(state).squeeze().detach().cpu().numpy())

    return action


replay = []
episods_rewards = []
render = True
max_replay = 100000
current_ite = 1
frequency_update_model = 10

n_episodes = 20
episode_len = 100000

epsilon = 0.1

env = gym.make('Breakout-v0')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(n=env.action_space.n).to(device)

for i_episode in range(n_episodes):

    observations = []
    rewards = []

    observation = env.reset()

    observations.append(observation)

    for t in range(episode_len):
        current_ite += 1
        if render:
            env.render()

        state = get_state(observations)

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = predict_action(model, state)

        observation, reward, done, info = env.step(action)

        replay.append((state, action, reward, done))
        observations.append(observation)
        rewards.append(reward)

    episods_rewards.append(np.sum(rewards))

    replay = replay[-max_replay:]

    del observations

env.close()
