#!/usr/bin/env python3

import time
import gym
import numpy as np
import matplotlib.pyplot as plt
import torch as th

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import DDPG
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from gym_pybullet_drones.utils.Logger import Logger

from HA import HoverAviary

freq = 50

env = HoverAviary(gui = False, record = False, freq = freq)

print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)

n_actions = env.action_space.shape[-1]

action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

env = Monitor(env)
# policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[512,512,256,128])
# model = DDPG(td3ddpgMlpPolicy, env, action_noise = action_noise, verbose = 1, policy_kwargs=policy_kwargs)
# model = DDPG.load("HA_agent_1.zip", action_noise = action_noise, env = env)
# model = DDPG.load("HA_agent_2.zip", action_noise = action_noise, env = env)
model = DDPG.load("HA_agent_1L.zip", action_noise = action_noise, env = env)
# model = DDPG.load("HA_agent_2L.zip", action_noise = action_noise, env = env)
n_ep = 1000
ep_len = int(freq*5.2)

for _ in range(1):
	model.learn(ep_len*n_ep, eval_freq = 10)

	new_save = False
	if new_save:
		model.save("HA_agent_2L")
	else:
		model.save("HA_agent_1L")

#print(env.get_episode_rewards())

plt.plot([i for i in range(len(env.get_episode_rewards()))],env.get_episode_rewards())
plt.show()

env = HoverAviary(gui = True, record = False, freq = 50)

obs = env.reset()
rew = []

logger = Logger(logging_freq_hz=int(1),
                num_drones=1)

for i in range(5):
	done = False
	env.reset()
	tot = 0
	step = 1
	while not done:
		action, _state = model.predict(obs, deterministic = True)
		obs, reward, done, _= env.step(action)
		tot += reward
		print(step)
		step += 1
		logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]),
                   control=np.zeros(12))
	rew.append(tot)
print(rew)
logger.plot()
env.close()