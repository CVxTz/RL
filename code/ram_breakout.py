import json
import random

import cv2
import gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class Model(torch.nn.Module):
    def __init__(self, n=4, channels=4):
        super(Model, self).__init__()

        self.embed = torch.nn.Embedding(256, 16)

        self.fc1 = torch.nn.Linear(channels * 128 * 16, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)

        self.fc4 = torch.nn.Linear(128, n)

        self.l_relu = torch.nn.LeakyReLU(0.1)

    def forward(self, x):
        x = self.embed(x)

        x = torch.flatten(x, start_dim=1)

        x = self.l_relu(self.fc1(x))
        x = self.l_relu(self.fc2(x))
        x = self.l_relu(self.fc3(x))

        y = self.fc4(x)

        return y


def get_state(l_obs, frames=4):
    new_l_obs = l_obs[-frames:]
    while len(new_l_obs) < frames:
        new_l_obs.append(new_l_obs[-1])

    state = np.concatenate(new_l_obs).astype(int)

    return state


def save_rewards(episods_rewards, out_path="../logs/episodes_rewards.json"):
    with open(out_path, "w") as f:
        json.dump(episods_rewards, f, indent=4)


def predict_action(model, state, device):
    model.eval()
    state = torch.from_numpy(state).long()
    state = state.to(device)

    state = state.unsqueeze(0)

    action = np.argmax(model(state).squeeze().detach().cpu().numpy())

    return action


def train_on_batch(
    model, optimizer, device, replay, batch_size=32, gamma=0.99, current_it=0, clip=2.0
):
    model.train()

    indexes = np.random.randint(0, len(replay), batch_size).tolist()
    batch = [replay[idx] for idx in indexes]
    states = np.array([a[0] for a in batch])
    future_states = np.array([a[1] for a in batch])
    actions = np.array([a[2] for a in batch])
    rewards = np.array([a[3] for a in batch])
    done = np.array([0 if a[4] else 1 for a in batch])

    states = torch.from_numpy(states).long()
    future_states = torch.from_numpy(future_states).long()

    actions = torch.from_numpy(actions).long()
    rewards = torch.from_numpy(rewards).float()
    done = torch.from_numpy(done).float()

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
    loss = F.mse_loss(q_s_a, y)

    if current_it % 5000 == 0:
        print("loss", loss.float(), "q_s_a", q_s_a[0].float())

    loss.backward()

    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

    optimizer.step()


if __name__ == "__main__":

    replay = []
    episods_rewards = []
    render = False
    max_replay = 1000000
    update_target_weight_freq = 10000
    current_ite = 1

    batch_size = 32

    frequency_update_model = 4

    n_episodes = 1000000
    episode_len = 10000

    epsilon_min = 0.1

    model_path = "../logs/ram_model.pkl"
    json_path = "../logs/ram_episodes_rewards.json"

    env = gym.make("Breakout-ram-v0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(n=env.action_space.n).to(device)
    target_model = Model(n=env.action_space.n).to(device)

    try:
        model.load_state_dict(torch.load(model_path))
        target_model.load_state_dict(torch.load(model_path))
    except:
        print("No model to load")

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for i_episode in tqdm(range(n_episodes)):

        epsilon = epsilon_min

        observations = []
        rewards = []

        observation = env.reset()

        observations.append(observation)

        for t in tqdm(range(episode_len)):
            current_ite += 1
            if render:
                env.render()

            state = get_state(observations)

            if (
                random.uniform(0, 1) < epsilon
                or current_ite < update_target_weight_freq + 10
            ):
                action = env.action_space.sample()
            else:
                action = predict_action(target_model, state, device)

            observation, reward, done, info = env.step(action)

            observations.append(observation)

            future_state = get_state(observations)

            replay.append((state, future_state, action, reward, done))

            rewards.append(reward)

            if done:
                break

            if current_ite % update_target_weight_freq == 0:
                target_model.load_state_dict(model.state_dict())

            if current_ite > batch_size and current_ite % frequency_update_model == 0:
                train_on_batch(
                    model,
                    optimizer,
                    device,
                    replay,
                    batch_size=batch_size,
                    gamma=0.99,
                    current_it=current_ite,
                )

        episods_rewards.append(np.sum(rewards))

        if i_episode % 100 == 0:
            save_rewards(episods_rewards, out_path=json_path)
            torch.save(model.state_dict(), model_path)

        replay = [a for a in replay if random.uniform(0, 1) < 0.7]

        del observations
        del rewards

    env.close()
