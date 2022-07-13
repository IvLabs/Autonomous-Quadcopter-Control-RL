import gym
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt
import torch as th
import numpy as np
import pickle

env = HoverAviary(gui=False,record=False)
env = Monitor(env,'monitor_name')
eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=75000,
                             render=False)
# env.reset()

# Customising Neural Network architecture
policy_kwargs = dict(activation_fn=th.nn.ReLU,net_arch=[512,512,256,128])

model = PPO('MlpPolicy',env,policy_kwargs=policy_kwargs,verbose=1,device='cuda')

# model = PPO.load("ppo_hover_2203_02", env=env)

model.learn(3000000,callback=eval_callback)

t = env.get_episode_rewards()
model.save("drone_model")
del model




file_name = "rewards_val.pkl"
op_file = open(file_name,'wb')
pickle.dump(t, op_file)
op_file.close()

fi,a = plt.subplots()
a.plot(np.arange(len(t)),t)
plt.show()
