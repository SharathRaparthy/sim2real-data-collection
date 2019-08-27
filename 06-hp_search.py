import numpy as np
import random
import os
from gym_ergojr.sim.single_robot import SingleRobot
from scripts.goal_babbling import GoalBabbling
import matplotlib.pyplot as plt

seed = 225
random.seed(seed)
np.random.seed(seed=seed)
total_steps = 200 * 100
rest_interval = 10 * 100
freq = 10
count = 0
steps_until_resample = 100/freq

#HYPERPARAMETERS
SAMPLE_NEW_GOAL = 1
NUMBER_OF_RETRIES = [5, 10, 20]
ACTION_NOISE = [0.1, 0.2, 0.4]
K_NEAREST_NEIGHBOURS = 8
EPSILON = [0.1, 0.2, 0.3]
file_path = os.getcwd() + '/files'
# os.makedirs(file_path + f'/{seed}')
for action_noise in ACTION_NOISE:
    for num_retries in NUMBER_OF_RETRIES:
        for epsilon in EPSILON:
            goal_babbling = GoalBabbling(action_noise, num_retries)

            # Reset the robot
            robot = SingleRobot(debug=False)
            robot.reset()
            robot.step()

            end_pos = []
            history = []
            goal_positions = []
            count = 0

            for epi in range(total_steps):
                if epi % rest_interval == 0:
                    print(f'Taking Rest at {epi}')
                    robot.reset()
                    robot.step()

                if epi % steps_until_resample == 0:
                    goal = [random.uniform(-0.1436, 0.22358), random.uniform(0.016000, 0.25002)]
                    if count < 10:
                        action = goal_babbling.sample_action()
                    else:
                        action = goal_babbling.sample_action() if random.random() < epsilon \
                            else goal_babbling.action_retries(goal, history)
                    count += 1
                _, end_position = goal_babbling.perform_action(action)
                history.append((action, end_position))
                end_pos.append(end_position)
                goal_positions.append(goal)

            final_pos = np.asarray(end_pos)
            final_goals = np.asarray(goal_positions)
            np.savez(file_path + f'/{seed}' + f'/numpy_files/pos-action-noise-{action_noise}-retries-{num_retries}-eps-{epsilon}.npz',
                     position=final_pos, goals=final_goals)
            plt.hist2d(final_pos[:, 0], final_pos[:, 1], bins=100)
            plt.xlim(-0.1436, 0.22358)
            plt.ylim(0.016000, 0.25002)
            plt.title(label=f'Action Noise - {action_noise} | Num retries - {num_retries} | Epsilon : {epsilon}')
            plt.savefig(file_path + f'/{seed}' + '/plots/hm-noise-{}-retries-{}-eps-{}.png'.format(action_noise, num_retries, epsilon))

