# Drone Obstacle Avoidance using RL

This project attempts to use Reinforcement Learning to train a model to perform simple maneuvers, plan navigation and avoid dynamic obstacles

<p align="center">
<img src="https://i.imgur.com/gmqkQMU.gif" width="400" height="300" align="Center">
</p>

![](https://i.imgur.com/WA8DUG9.png)

## Current State of the Project

Training the Model to perform simple maneuvers

## Environment and Model

### Environment

The environment used is [Pybullet Gym Environment](https://github.com/utiasDSL/gym-pybullet-drones) for quadrotors.

The base class for all 'drone aviary' environments is defined under [``BaseAviary.py``](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/BaseAviary.py).
The file loads Drone Model, Physics Simulator, and defines various Environment parameters and loads parameters from URDF files, to render the simulation

[``BaseSingleAgentAviary.py``](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/single_agent_rl/BaseSingleAgentAviary.py) is a subclass of BaseAviary, dedicated for all single drone environments for RL. The action space and observation space are defined here, also a function is defined to compute current observation of the environment.

There are 6 different available Action Types
- RPM - Desired rpm values for each propeller
- DYN - Desired thrust and torque values for each propeller
- PID - PID controller
- ONE_D_RPM - Identical input rpm value to all propellers
- ONE_D_DYN - Identical thrust/torque to all propellers
- ONE_D_PID - Identical PID controller for all propellers

While training our agent, we have used the default action type, ``ActionType.RPM``.

Also, there are 2 different Observation Types
- KIN - Kinematic information (position, linear and angular velocities)
- RGB - camera capture of each drone's Point of View

``ObservationType.KIN``, was used in training our agent

The above class is then used to construct four single agent RL problems:
- [TakeOffAviary](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/single_agent_rl/TakeoffAviary.py) : Single agent RL problem: take-off
- [HoverAviary](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/single_agent_rl/HoverAviary.py) : Single agent RL problem: hover at position
- [FlyThruGateAviary](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/single_agent_rl/FlyThruGateAviary.py) : Single agent RL problem: fly through a gate
- [TuneAviary](https://github.com/utiasDSL/gym-pybullet-drones/blob/master/gym_pybullet_drones/envs/single_agent_rl/TuneAviary.py) : Single agent RL problem: optimize PID coefficients

We trained various agents extensively on HoverAviary, and intend to discuss various problems we faced, and our progress through this repo.

### Model

The models used in this project are from [``Stable Baseline3``](https://stable-baselines3.readthedocs.io/en/master/). We tried a variety of RL algorithms to train models for all the tasks. Up till now, we've used two: Deep Deterministic Policy Gradient (DDPG) and Proximal Policy Optimization (PPO).

#### DDPG

While using DDPG to train our model, we found that training the model using DDPG took a lot of time, i.e., it processed through the training episodes at a slow pace, and the results didn't look too promising.

So we shifted to PPO.

#### PPO

We found that PPO had a better training speed than DDPG for the same number of time steps in the environment. The results weren't encouraging. So we increased the network size of the model from ``[32, 32]`` to ``[512, 512, 256, 128]`` and the larger neural network showed better results

## Tasks

### Hovering

In this task, the drone starts from a position on the ground. The agent then is supposed to cause the drone to take-off and hover at a target location.

The environment for this task was supposed to be `HoverAviary`. The reward function used for this environment didn't seem rich enough, so we made certain modifications, as

#### Eucledian Distance

The default reward function in 'HoverAviary' generated the reward using euclidian distance as follows:

```python=
reward = -1 * np.linalg.norm(np.array([0, 0, 1])-state[0:3])**2 #here state[0:3] denotes the current position of the drone
```

#### Error Sphere

Euclidean distance alone was not able to ensure that the agent moved towards the target location. So we added error spheres which basically give a different magnitude of rewards based on distance. This was done to ensure that the agent did not stray too far from the target location and maintained a stable hover near it.

<p align="center">
<img src="https://i.imgur.com/fnIMMNH.gif" width="300" height="300" align="Center">
</p>

#### Reward based on Rotation

The reward functions up till now only rewarded reaching the target location and were not dependent on the roll, pitch, or yaw. So we picked certain value ranges for the roll, pitch, yaw, and their respective time derivatives, and returned negative when they exceeded these value ranges.

#### Exponential and Euclidean Distance Rewards

The Error Sphere idea was not too successful in training the agent to move towards the target so we returned rewards by taking the exponential of the euclidean distance.

<p align="center">
<img src="https://i.imgur.com/VBOywe3.gif" width="300" height="300" align="Center">
</p>

![](https://i.imgur.com/IeqSdOl.png)


### Results with Action Space ONE_D_RPM
- We changed the action space of the environment to apply same action (RPM Value)to all four propellers.
- After a training of 3,250,000 Million steps, the model was able to learn the task of hovering

#### Results and Simulation

<p align="center">
<img src="https://i.imgur.com/YDinZwy.png" width="600" height="450" align="Center">
</p>

<p align="center">
<img src="https://i.imgur.com/gmqkQMU.gif" width="400" height="300" align="Center">
</p>

<p align="center">
<img src="https://i.imgur.com/yMnckDg.png" width="600" height="450" align="Center">
</p>