#!/usr/bin/env python3

import time
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import DDPG
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

EPISODE_REWARD_THRESHOLD = -0

use_saved = True
new_save = False

env = HoverAviary(gui = False, record = False)
print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)

n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

env = Monitor(env)

if ((not use_saved)):
	model = DDPG(td3ddpgMlpPolicy, env, action_noise = action_noise, verbose = 1)
else:
	model = DDPG.load("agent_1.zip", action_noise = action_noise, env = env)

n_ep = 500

model.learn(1200*n_ep, eval_freq = 2)

if new_save:
	model.save("agent_2")
else:
	model.save("agent_1")

print(env.get_episode_rewards())

plt.plot([i for i in range(len(env.get_episode_rewards()))],env.get_episode_rewards())
plt.show()

env = HoverAviary(gui = True, record = False)
obs = env.reset()
rew = []

for i in range(2):
	done = False
	env.reset()
	tot = 0
	while not done:
		action, _state = model.predict(obs, deterministic = True)
		obs, reward, done, _= env.step(action)
		tot += reward
	rew.append(tot)
print(rew)