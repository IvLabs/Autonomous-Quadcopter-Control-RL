#!/usr/bin/env python3

import os
import time
import argparse
from sys import platform
import argparse
import subprocess
import pdb
import math
import random
import gym
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import torch

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env # Module cmd_util will be renamed to env_util https://github.com/DLR-RM/stable-baselines3/pull/197
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import DDPG
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy

from gym_pybullet_drones.envs.single_agent_rl.TakeoffAviary import TakeoffAviary
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary

#import shared_constants

EPISODE_REWARD_THRESHOLD = -0

print("hi")

env_name = 0

#sa_env_kwargs = dict(aggregate_phy_steps=1, obs="kin", act="one_d_rpm")
sa_env_kwargs = dict(aggregate_phy_steps=1)

if env_name:
        train_env = make_vec_env(TakeoffAviary,
                                 env_kwargs=sa_env_kwargs,
                                 n_envs=1,
                                 seed=0
                                 )
        env_name = "takeoff-aviary-v0"
else:
        train_env = make_vec_env(HoverAviary,
                                 env_kwargs=sa_env_kwargs,
                                 envs=1,
                                 seed=0)
        env_name = "hover-aviary-v0"

print("[INFO] Action space:", train_env.action_space)
print("[INFO] Observation space:", train_env.observation_space)

offpolicy_kwargs = dict(activation_fn=torch.nn.ReLU,
                            net_arch=[512, 512, 256, 128])

model = DDPG(td3ddpgMlpPolicy,
                    train_env,
                    policy_kwargs=offpolicy_kwargs,
                    verbose=1)

eval_env = gym.make(env_name,
                            aggregate_phy_steps=1)

#eval_env = VecTransposeImage(eval_env)

#print(train_env.obs)

callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=EPISODE_REWARD_THRESHOLD,
                                                     verbose=1
                                                     )
eval_callback = EvalCallback(eval_env,
                                 callback_on_new_best=callback_on_best,
                                 verbose=1,
                                 eval_freq=2000,
                                 deterministic=True,
                                 render=True
                                 )
model.learn(total_timesteps=35000, #int(1e12),
                callback=eval_callback,
                log_interval=100)