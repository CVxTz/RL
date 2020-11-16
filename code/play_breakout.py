import gym
import torch
from tqdm import tqdm
from pixel_breakout import Model, resize, to_gray, get_state, predict_action
import time

if __name__ == "__main__":

    render = True
    current_ite = 1

    n_episodes = 1000000
    episode_len = 10000

    model_path = "../logs/pixel_model.pkl"

    env = gym.make("Breakout-v0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target_model = Model(n=env.action_space.n).to(device)

    target_model.load_state_dict(torch.load(model_path))

    for i_episode in tqdm(range(n_episodes)):

        observations = []
        rewards = []

        observation = env.reset()

        observations.append(resize(to_gray(observation)))

        for t in tqdm(range(episode_len)):
            time.sleep(0.05)
            current_ite += 1
            if render:
                env.render()

            state = get_state(observations)

            action = predict_action(target_model, state, device)

            observation, reward, done, info = env.step(action)

            observations.append(resize(to_gray(observation)))

            future_state = get_state(observations)

            # cv2.imwrite("../logs/img_%s.jpg" % current_ite, future_state)

            rewards.append(reward)

            if done:
                break

        del observations
        del rewards

    env.close()
