import time
import os
import numpy as np
from gym_ergojr.sim.abstract_robot import PusherRobot
from arguments import get_args

robot = PusherRobot(action_noise=False, obs_noise=True, debug=False) # 3DOF Pusher
args = get_args()

file_path = os.getcwd() + '/data/{}/freq{}/{}/'.format(args.env_name, args.freq, args.approach)

rest_interval = 10 * 100
freq = args.freq
steps_until_resample = 100/freq

actions_trajectories = np.load(file_path + 'action_trajectories.npz')
actions = actions_trajectories["actions"]
real_trajectories = np.zeros((actions.shape[0], 6))

for epi in range(actions.shape[0]):
    start = time.time()

    if epi % rest_interval == 0:
        print("Episodes completed: {}".format(epi))
        robot.hard_reset()
        robot.rest()
    action = actions[epi, :]
    robot.act(action)
    robot.step()
    obs = robot.observe()

    real_trajectories[epi, :] = obs

    delta = start - time.time()
    #
    # if delta < 0.01:
    #     time.sleep(0.01 - delta)


np.save(file_path + 'real_world_trajectories_obs_noise.npy', real_trajectories)








