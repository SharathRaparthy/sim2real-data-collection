import numpy as np
import os
from gym_ergojr.sim.abstract_robot import PusherRobot
from arguments import get_args
from gym_ergojr.sim.objects import Puck


args = get_args()
robot = PusherRobot(action_noise=args.action_noise, obs_noise=args.obs_noise, debug=False) # 3DOF Pusher
# include puck
puck = Puck()
file_path = os.getcwd() + '/data/{}/freq{}/{}/'.format(args.env_name, args.freq, args.approach)

if not os.path.isdir(file_path):
    os.makedirs(file_path)

np.random.seed(seed=args.seed)
total_steps = args.total_steps
rest_interval = args.rest_interval
freq = args.freq
count = 0
steps_until_resample = 100/freq

sim_trajectories = np.zeros((total_steps, 8)) # Dont hard code it.
actions = np.zeros((total_steps, 3))  # Don't hard code it.
bad_actions = np.zeros((total_steps))
robot.hard_reset()
robot.rest()
puck.hard_reset()
robot.step()
end_pos = []

for epi in range(total_steps):

    if epi % rest_interval == 0:
        print('Taking Rest at {}'.format(epi))
        robot.hard_reset()
        puck.hard_reset()
        robot.rest()
        robot.step()

    if epi % steps_until_resample == 0:
        action = np.random.uniform(-1, 1, 3)
    end_pos.append(robot.get_tip())
    actions[epi, :] = action
    robot.act(actions[epi, :])
    robot.step()
    robot_obs = robot.observe()
    puck_obs = puck.normalize_puck()
    obs = np.hstack([robot_obs, puck_obs])
    sim_trajectories[epi, :] = obs

np.save(file_path + 'end_positions.npy', end_pos)
np.savez(file_path + '/action_trajectories.npz', actions=actions, trajectories=sim_trajectories)

