import numpy as np
import gym
import time
from gym_ergojr.sim.single_robot import SingleRobot
from create_actions import Actions

robot = SingleRobot(debug=False)


file_path = '/home/sharath/sim2real-record/data/freq10/'


total_steps = 2000 * 100
rest_interval = 10 * 100
freq = 10
count = 0
steps_until_resample = 100/freq

sim_trajectories = np.zeros((total_steps, 12))
actions = np.zeros((total_steps, 6))
bad_actions = np.zeros((total_steps))
robot.reset()
robot.step()

for epi in range(total_steps):

    if epi % rest_interval == 0:
        print(f'Taking Rest at {epi}')
        robot.reset()
        robot.step()

    if epi % steps_until_resample == 0:
        action = np.random.uniform(-1, 1, 6)
        action[:][0], action[:][3] = 0, 0

    actions[epi, :] = action
    robot.act2(actions[epi, :])
    robot.step()
    obs = robot.observe()
    sim_trajectories[epi, :] = obs

    if len(robot.get_hits(robot1=0, robot2=None)) > 1:  # check for bad actions
        count += 1
        actions[epi, :] = 0
        sim_trajectories[epi, :] = 0

print(count)

actions = actions[bad_actions == 0]
sim_trajectories = sim_trajectories[bad_actions == 0]
np.savez(file_path + '/04-clean_action_trajectories.npz', actions=actions, trajectories=sim_trajectories)

