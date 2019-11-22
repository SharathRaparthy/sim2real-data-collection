import numpy as np
import random
import os
from arguments import get_args
from scripts.goal_babbling import GoalBabbling
import matplotlib.pyplot as plt
from gym_ergojr.sim.objects import Puck

seed = 123
args = get_args()

random.seed(seed)
np.random.seed(seed=seed)
total_steps = 1000 * 100
rest_interval = 10 * 100
freq = args.freq
count = 0
steps_until_resample = 100/freq
max_history_len = 15000
puck = Puck()

# HYPERPARAMETERS
SAMPLE_NEW_GOAL = 1
NUMBER_OF_RETRIES = [5, 10, 20]
ACTION_NOISE = [0.1, 0.2, 0.4]
K_NEAREST_NEIGHBOURS = 8
EPSILON = [0.1, 0.2, 0.3]
file_path = os.getcwd() + '/files/{}'.format(args.env_name)
if not os.path.isdir(file_path):
    os.makedirs(file_path + '/{}/numpy_files/'.format(seed))
    os.makedirs(file_path + '/{}/plots/'.format(seed))
task = 'pusher'
for action_noise in ACTION_NOISE:
    print('================================================')
    print('Approach is : {} | Task is {} | Frequency is : {}'.format(args.approach, task, args.freq))
    print('================================================')
    for num_retries in NUMBER_OF_RETRIES:
        for epsilon in EPSILON:

            goal_babbling = GoalBabbling(action_noise, num_retries, task)

            # Reset the robot
            goal_babbling.reset_robot()

            end_pos = []
            history = []
            goal_positions = []
            count = 0

            for epi in range(total_steps):
                if epi % rest_interval == 0:  # Reset the robot after every rest interval
                    print('Taking Rest at {}'.format(epi))
                    goal_babbling.reset_robot()

                if epi % steps_until_resample == 0:
                    # goal = [random.uniform(-0.1436, 0.22358), random.uniform(0.016000, 0.25002)]  # Reacher goals
                    # goal = [random.uniform(-0.135, 0.0), random.uniform(-0.081, 0.135)]  # Pusher goals
                    puck.hard_reset()
                    goal = puck.normalize_puck()
                    if count < 10:
                        action = goal_babbling.sample_action()
                    else:
                        action = goal_babbling.sample_action() if random.random() < epsilon \
                            else goal_babbling.action_retries(goal, history)
                    count += 1

                if task == 'reacher':
                    action[0], action[3] = 0, 0
                _, end_position, observation = goal_babbling.perform_action(
                    action)  # Perform the action and get the observation
                if len(history) >= max_history_len:
                    del history[0]
                history.append((action, end_position))  # Store the actions and end positions in buffer
                end_pos.append(end_position)
                goal_positions.append(goal)

            final_pos = np.asarray(end_pos)
            final_goals = np.asarray(goal_positions)
            np.savez(file_path + '/{}'.format(seed) +
                     '/numpy_files/pos-action-noise-{}-retries-{}-eps-{}.npz'.format(action_noise,
                                                                                     num_retries,
                                                                                     epsilon),
                     position=final_pos,
                     goals=final_goals)
            plt.hist2d(final_pos[:, 0], final_pos[:, 1], bins=100)
            # plt.xlim(-0.1436, 0.22358)
            # plt.ylim(0.016000, 0.25002)
            plt.title(label='Action Noise - {} | Num retries - {} | Epsilon : {}'.format(action_noise,
                                                                                       num_retries,
                                                                                        epsilon))
            plt.savefig(file_path + '/{}'.format(seed) + '/plots/hm-noise-{}-retries-{}-eps-{}.png'.format(action_noise,
                                                                                                           num_retries,
                                                                                                           epsilon))

