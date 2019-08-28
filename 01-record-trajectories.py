import numpy as np
import matplotlib.pyplot as plt
from gym_ergojr.sim.single_robot import SingleRobot

robot = SingleRobot(debug=False)


file_path = '/home/sharath/sim2real-data-collection/'
np.random.seed(seed=123)
total_steps = 300 * 100
rest_interval = 10 * 100
freq = 10
count = 0
steps_until_resample = 100/freq

sim_trajectories = np.zeros((total_steps, 12))
actions = np.zeros((total_steps, 6))
bad_actions = np.zeros((total_steps))
robot.reset()
robot.step()
end_pos = []
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
    end_position = robot.get_tip()[0][1:]
    end_pos.append(end_position)

    if len(robot.get_hits(robot1=0, robot2=None)) > 1:  # check for bad actions
        count += 1
        actions[epi, :] = 0
        sim_trajectories[epi, :] = 0
final_pos = np.asarray(end_pos)
end_pos_path = file_path + 'random_end_pos.npy'
np.save(end_pos_path, final_pos)
plt.scatter(final_pos[:, 0], final_pos[:, 1], alpha=0.5)
plt.show()
plt.hist2d(final_pos[:, 0], final_pos[:, 1], bins=100)
plt.show()
# actions = actions[bad_actions == 0]
# sim_trajectories = sim_trajectories[bad_actions == 0]
# np.savez(file_path + '/04-clean_action_trajectories.npz', actions=actions, trajectories=sim_trajectories)

