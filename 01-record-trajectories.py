import numpy as np
import matplotlib.pyplot as plt
from gym_ergojr.sim.single_robot import SingleRobot
from arguments import get_args
import os

robot = SingleRobot(debug=False) # 6DOF Reacher Robot
args = get_args()

file_path = os.getcwd() + '/data/freq{}/{}'.format(args.freq, args.approach)

if not os.path.isdir(file_path):
    os.makedirs(file_path)

np.random.seed(seed=225)
total_steps = 10000 * 100
rest_interval = 10 * 100
freq = args.freq
count = 0
steps_until_resample = 100/freq

sim_trajectories = np.zeros((total_steps, 12))
actions = np.zeros((total_steps, 6))
bad_actions = np.zeros((total_steps))
robot.reset()
robot.step()
end_pos = []

for epi in range(total_steps):

    if epi % rest_interval == 0:  # Take rest after every 10 * 100 steps
        print('Taking Rest at {}'.format(epi))
        robot.reset()
        robot.step()

    if epi % steps_until_resample == 0:  # Sample a new action after certain steps
        action = np.random.uniform(-1, 1, 6)
        action[:][0], action[:][3] = 0, 0  # Motor 0 and 3 fixed => 4DOF

    '''Perform action and record the observation and the tip position'''
    actions[epi, :] = action
    robot.act2(actions[epi, :])
    robot.step()
    obs = robot.observe()

    sim_trajectories[epi, :] = obs
    end_position = robot.get_tip()[0][1:]
    end_pos.append(end_position)

    if len(robot.get_hits(robot1=0, robot2=None)) > 1:  # check for bad actions
        count += 1
        actions[epi, :] = 0
        sim_trajectories[epi, :] = 0

final_pos = np.asarray(end_pos)
end_pos_path = file_path + '/random_end_pos_{}.npy'.format(args.freq)
np.save(end_pos_path, final_pos)
plt.scatter(final_pos[:, 0], final_pos[:, 1], alpha=0.5)
plt.show()
plt.hist2d(final_pos[:, 0], final_pos[:, 1], bins=100)
plt.show()
# np.savez(file_path + '/action_trajectories.npz', actions=actions, trajectories=sim_trajectories)

