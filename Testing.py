import gym
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from gym_pybullet_drones.utils.Logger import Logger
env = HoverAviary(gui=True,record=False)

model = PPO.load("best_model.zip", env=env,custom_objects={"learning_rate": 0.0,
            "lr_schedule": lambda _: 100.0,
            "clip_range": lambda _: 100.0,})


rew = []
logger = Logger(logging_freq_hz=int(1),
                num_drones=1)
obs = env.reset()
tot_rew = 0
done=False
while not done:
    action, _state = model.predict(obs)
    print(action)
    obs, reward, done, info = env.step(action)
    tot_rew+=reward
    rew.append(reward)
    logger.log(drone=0,
                   timestamp=1 / env.SIM_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]),
                   control=np.zeros(12))

    if done:
        obs = env.reset()

logger.plot()
print(tot_rew)

x_axis = np.arange(len(rew))

fig,ax = plt.subplots()
ax.plot(x_axis,rew)
plt.show()