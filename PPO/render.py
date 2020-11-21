import gym
import torch
from PIL import Image
from tqdm import tqdm
import random
import imageio
import cv2
import numpy as np


from PPO.model import (
    PolicyNetwork,
    ValueNetwork,
    device,
)


def write_on_image(img, reward):

    cv2.putText(
        img,
        f"Sum Reward: {int(reward)}",
        (0, img.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--policy_path")
    parser.add_argument("--env_name")
    parser.add_argument("--n_episodes", type=int, default=5)
    parser.add_argument("--max_timesteps", type=int, default=400)

    parser.add_argument("--out_gif")

    state_scale = 1.0

    args = parser.parse_args()

    policy_path = args.policy_path
    env_name = args.env_name

    n_episodes = args.n_episodes
    max_timesteps = args.max_timesteps

    out_gif = args.out_gif

    env = gym.make(env_name)
    observation = env.reset()
    n_actions = env.action_space.n
    feature_dim = observation.size

    policy_model = PolicyNetwork(in_dim=feature_dim, n=n_actions).to(device)

    policy_model.load_state_dict(torch.load(policy_path))

    frames = []
    sum_reward = 0
    for _ in tqdm(range(n_episodes)):
        observation = env.reset()

        for timestep in range(max_timesteps):

            frames.append(np.ascontiguousarray(env.render(mode="rgb_array")))

            write_on_image(frames[-1], sum_reward)

            action = policy_model.best_action(observation / state_scale)

            new_observation, reward, done, info = env.step(action)
            sum_reward += reward

            if done:
                for a in range(10):
                    frames.append(np.ascontiguousarray(env.render(mode="rgb_array")))
                    write_on_image(frames[-1], sum_reward)
                break

            observation = new_observation

    imageio.mimsave(out_gif, frames, fps=60)
