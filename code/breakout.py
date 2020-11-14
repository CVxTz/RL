import gym
import cv2
import random
import numpy as np


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


replay = []
render = True
max_replay = 10000

epsilon = 0.1

env = gym.make('Breakout-v0')


for i_episode in range(20):

    observations = []

    observation = env.reset()

    observations.append(observation)

    for t in range(100):

        if render: env.render()

        action = env.action_space.sample()

        observation, reward, done, info = env.step(action)

        replay.append((get_state(observations), action, reward, done))
        observations.append(observation)
        cv2.imwrite("../logs/%s.jpg" % ((t + 1) + 1000*(i_episode + 1)), replay[-1][0])

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

    replay = replay[-max_replay:]

    del observations

env.close()