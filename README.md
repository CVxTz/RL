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