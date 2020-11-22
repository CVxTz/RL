# RL

### PPO
How to run:
```
python PPO/ppo.py
```
Test Policy:
```
python PPO/render.py --policy_path logs/Lunar/LunarLander-v2_100_policy.pth \ 
                     --env_name LunarLander-v2 --out_gif logs/lunar_late.gif

```

### Learning to Play CartPole and LunarLander with Proximal Policy Optimization

#### Implementing PPO from scratch with Pytorch

![](https://cdn-images-1.medium.com/max/800/1*WiKzN5tiKqettn8yeLj-MQ.gif)

In this post, we will train an RL agent to play two control based games:

* [https://gym.openai.com/envs/LunarLander-v2/](https://gym.openai.com/envs/LunarLander-v2/)
* [https://gym.openai.com/envs/CartPole-v1/](https://gym.openai.com/envs/CartPole-v1/)

Our agent will be trained using an algorithm called Proximal Policy
Optimization. We will implement this approach from scratch using PyTorch and
OpenAi gym.

This project is based on the following paper:

* [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

#### Gym:

The basic idea behind OpenAI Gym is that we define an environment env by
calling:

    env = gym.make(env_name)

Then at each time step **t**, we pick an action **a** and we get a new
state_(t+1) and a reward **reward_t**. The objective is to train an Agent that
learns a policy PI that can predict for each state the best action that will
maximize the sum of the future rewards. For example, in the environment
LunarLander, we get the maximum reward if we land the rocket smoothly on top of
the landing area. In the environment CartPole, the objective is to keep the pole
vertical for as long as possible.

![](https://cdn-images-1.medium.com/max/800/1*RxpfmLGwZR8kEVOlxbSjzQ.gif)

#### PPO:

Our final objective is to learn a policy network that will take the state as
input and then output a probability distribution over the actions that will
maximize the expected reward.

Implementing PPO goes as follows:

* First, we start with a policy PI_old
* We sample some trajectories from P_old
* For each action **a** in each trajectory we compute the Advantage, a kind of
measure of how much better the action **a** is compared to other possible
actions at state_t.
* For a few epochs, we maximize the following objective with gradient ascent:

![](https://cdn-images-1.medium.com/max/800/1*cnjyaLHg0QynODhIXkYYSg.png)

<span class="figcaption_hack">From [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)</span>

What this loss does is that it **increases the probability** if action a_t at
state s_t if it has a **positive advantage** and **decreases the probability**
in the case of a **negative advantage**. However, in practice this ratio of
probabilities tends to diverge to infinity, making the training unstable. The
authors propose a clipped version of the loss to solve this issue:

![](https://cdn-images-1.medium.com/max/800/1*ocve-gRQDzkXVov-yTtZuA.png)

<span class="figcaption_hack">[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)</span>

We also add two additional terms to the loss, an mean squared error over the
Value function and Entropy encourage exploration during the sampling of the
trajectories.

We can plot the sum of the rewards during the progression of the training for
each episode:

![](https://cdn-images-1.medium.com/max/1200/1*MIF6DYmlJT9yLQFDafmOJA.png)

<span class="figcaption_hack">Training reward for LunarLander</span>

#### Trained Agents:

Now we get the see the trained policy network in action.

* CartPole

![](https://cdn-images-1.medium.com/max/800/1*_ddmwllJuY-9Zvh8x6PCbQ.gif)

* LunarLander

![](https://cdn-images-1.medium.com/max/800/1*tKbe-gnp6VujnrQ2YwEZ-g.gif)

Perfect Landing!

#### Other resources:

If you are interested in learning more about PPO or Policy gradient
Reinforcement Learning methods I recommend following this course:
[https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wj9ByIVpbfGc](https://www.youtube.com/playlist?list=PL_iWQOsE6TfURIIhCrlt-wj9ByIVpbfGc)
by Sergey Levine at Berkeley. The course is very long and math-heavy but the
instructor is really good.

#### Code :

[https://github.com/CVxTz/RL](https://github.com/CVxTz/RL)
