import numpy as np
import os
from gym_ergojr.sim.abstract_robot import PusherRobot
from arguments import get_args

robot = PusherRobot(action_noise=False, obs_noise=False, debug=False) # 3DOF Pusher
args = get_args()

file_path = os.getcwd() + '/data/{}/freq{}/{}/'.format(args.env_name, args.freq, args.approach)

if not os.path.isdir(file_path):
    os.makedirs(file_path)

np.random.seed(seed=123)
total_steps = 10000 * 100
rest_interval = 10 * 100
freq = args.freq
count = 0
steps_until_resample = 100/freq

sim_trajectories = np.zeros((total_steps, 6))
actions = np.zeros((total_steps, 3))
bad_actions = np.zeros((total_steps))
robot.hard_reset()
robot.rest()
robot.step()
end_pos = []
for epi in range(total_steps):

    if epi % rest_interval == 0:
        print('Taking Rest at {}'.format(epi))
        robot.hard_reset()
        robot.rest()
        robot.step()

    if epi % steps_until_resample == 0:
        action = np.random.uniform(-1, 1, 3)

    actions[epi, :] = action
    robot.act(actions[epi, :])
    robot.step()
    obs = robot.observe()

    sim_trajectories[epi, :] = obs

np.savez(file_path + '/action_trajectories.npz', actions=actions, trajectories=sim_trajectories)

