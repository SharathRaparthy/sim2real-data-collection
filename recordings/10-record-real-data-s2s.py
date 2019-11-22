import time
import os
import numpy as np
from gym_ergojr.sim.abstract_robot import PusherRobot
from gym_ergojr.sim.objects import Puck
from arguments import get_args
from common.envs import PusherRobotNoisy
from common.envs import NewPuck

args = get_args()
robot = PusherRobotNoisy(action_noise=args.action_noise, obs_noise=args.obs_noise) # 3DOF Pusher
puck = NewPuck()

file_path = os.getcwd() + '/data/{}/freq{}/{}/'.format(args.env_name, args.freq, args.approach)

rest_interval = 10 * 100
freq = args.freq
steps_until_resample = 100/freq

actions_trajectories = np.load(file_path + 'actions_trajectories.npz')
print([keys for keys in actions_trajectories.keys()])
actions = actions_trajectories["actions"]
trajectories = actions_trajectories["sim_trajectories"]
real_trajectories = np.zeros((trajectories.shape[0], trajectories.shape[1]))

for epi in range(actions.shape[0]):
    start = time.time()

    if epi % rest_interval == 0:
        print("Episodes completed: {}".format(epi))
        robot.hard_reset()
        puck.hard_reset()
        robot.rest()
    action = actions[epi, :]
    robot.act(action)
    robot.step()
    robot_obs = robot.observe()
    puck_obs = puck.normalize_puck()
    obs = np.hstack([robot_obs, puck_obs])
    real_trajectories[epi, :] = obs

    delta = start - time.time()

    # if delta < 0.01:
    #     time.sleep(0.01 - delta)


if args.obs_noise:
    np.save(file_path + 'real_world_trajectories_obs_noise.npy', real_trajectories)
else:
    np.save(file_path + 'real_world_trajectories_action_noise.npy', real_trajectories)








