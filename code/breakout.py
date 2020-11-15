import json
import random

import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from tqdm import tqdm


class Model(torch.nn.Module):
    def __init__(self, n=4, channels=3):
        super(Model, self).__init__()
        self.cnn = torch.nn.Conv2d(channels, 3, kernel_size=3)
        self.resnet = models.resnet18(pretrained=False)
        self.fc = torch.nn.Linear(1000, n)

    def forward(self, x):
        x = self.cnn(x)
        x = self.resnet(x)
        y = self.fc(x)

        return y


def to_gray(obs):
    return obs.mean(axis=-1)


def get_state(l_obs, frames=3):
    new_l_obs = l_obs[-frames:]
    while len(new_l_obs) < frames:
        new_l_obs.append(new_l_obs[-1])

    state = np.zeros(new_l_obs[-1].shape + (frames,))

    for i, obs in enumerate(new_l_obs):
        state[..., i] = obs

    return state


def save_rewards(episods_rewards, out_path="../logs/episodes_rewards.json"):
    with open(out_path, "w") as f:
        json.dump(episods_rewards, f, indent=4)


def predict_action(model, state):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.from_numpy(state).float()
    state = state.to(device)
    state = state.permute(2, 0, 1)

    state = state.unsqueeze(0)

    action = np.argmax(model(state).squeeze().detach().cpu().numpy())

    return action


def train_on_batch(model, optimizer, replay, batch_size=32, gamma=0.99, current_it=0):
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    indexes = np.random.randint(0, len(replay), batch_size).tolist()
    batch = [replay[idx] for idx in indexes]
    states = np.array([a[0] for a in batch])
    future_states = np.array([a[1] for a in batch])

    actions = np.array([a[2] for a in batch])
    rewards = np.array([a[3] for a in batch])
    done = np.array([0 if a[4] else 1 for a in batch])

    states = torch.from_numpy(states).float()
    future_states = torch.from_numpy(future_states).float()

    actions = torch.from_numpy(actions).long()
    rewards = torch.from_numpy(rewards).float()
    done = torch.from_numpy(done).float()

    states = states.permute(0, 3, 1, 2)
    future_states = future_states.permute(0, 3, 1, 2)

    states = states.to(device)
    future_states = future_states.to(device)

    actions = actions.to(device)
    rewards = rewards.to(device)
    done = done.to(device)

    q_s = model(states)
    q_s_prime = model(future_states)

    q_s_a = q_s.gather(1, actions.unsqueeze(1)).squeeze(1)
    q_s_prime_a, _ = torch.max(q_s_prime, dim=1)
    q_s_prime_a = q_s_prime_a.detach()

    y = rewards + gamma * done * q_s_prime_a

    optimizer.zero_grad()
    loss = F.smooth_l1_loss(q_s_a, y)

    if current_it % 1000 == 0:
        print("loss", loss.float(), "q_s_a", q_s_a[0].float())

    loss.backward()
    optimizer.step()


replay = []
episods_rewards = []
render = False
max_replay = 30000
update_target_weight_freq = 2000
current_ite = 1

frequency_update_model = 5

n_episodes = 10000
episode_len = 10000

epsilon = 0.1

env = gym.make('Breakout-v0')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(n=env.action_space.n).to(device)
target_model = Model(n=env.action_space.n).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

for i_episode in tqdm(range(n_episodes)):

    observations = []
    rewards = []

    observation = env.reset()

    observations.append(to_gray(observation))

    for t in tqdm(range(episode_len)):
        current_ite += 1
        if render:
            env.render()

        state = get_state(observations)

        if random.uniform(0, 1) < epsilon or current_ite < 2 * update_target_weight_freq:
            action = env.action_space.sample()
        else:
            action = predict_action(target_model, state)

        observation, reward, done, info = env.step(action)

        observations.append(to_gray(observation))
        future_state = get_state(observations)

        replay.append((state, future_state, action, reward, done))

        rewards.append(reward)

        if done:
            break

        if current_ite % update_target_weight_freq == 0:
            target_model.load_state_dict(model.state_dict())

        if current_ite > 32 and current_ite % frequency_update_model == 0:
            train_on_batch(model, optimizer, replay, batch_size=32, gamma=0.99, current_it=current_ite)

    episods_rewards.append(np.sum(rewards))

    if i_episode % 10 == 0:
        save_rewards(episods_rewards, out_path="../logs/episodes_rewards.json")
        torch.save(model.state_dict(), "../logs/model.pkl")

    replay = replay[-max_replay:]

    del observations
    del rewards

env.close()
