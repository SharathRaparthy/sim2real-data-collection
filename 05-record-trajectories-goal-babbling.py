import numpy as np
import random
from gym_ergojr.sim.single_robot import SingleRobot
from scripts.goal_babbling import GoalBabbling
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
robot = SingleRobot(debug=False)

# file_path = '/home/sharath/sim2real-record/data/freq10/'

total_steps = 400 * 100
rest_interval = 10 * 100
freq = 1
count = 0
steps_until_resample = 100/freq

#HYPERPARAMETERS
SAMPLE_NEW_GOAL = 1
NUMBER_OF_RETRIES = 1
ACTION_NOISE = 0.1
K_NEAREST_NEIGHBOURS = 8
EPSILON = 0.2

goal_babbling = GoalBabbling(ACTION_NOISE, NUMBER_OF_RETRIES)

robot.reset()
robot.step()
history = []
action = np.random.uniform(-1, 1, 6)
action[0], action[3] = 0, 0
robot.act2(action)
robot.step()
achieved_goal = robot.get_tip()[0][1:]
history.append((action, achieved_goal))
end_pos = []
goals = []
for epi in range(total_steps):
    if epi % SAMPLE_NEW_GOAL == 0:
        goal = [random.uniform(-0.12, 0.12), random.uniform(0.02, 0.24)]
    if epi % rest_interval == 0:
        print(f'Taking Rest at {epi}')
        robot.reset()
        robot.step()
    if random.random() < EPSILON:
        action = np.random.uniform(-1, 1, 6)
    else:
        action = goal_babbling.inverse(goal, history)
    action[0], action[3] = 0, 0
    robot.act2(action)
    robot.step()
    end_position = robot.get_tip()[0][1:]
    history.append((action, end_position))
    end_pos.append(end_position)

final_pos = np.asarray(end_pos)
plt.hist2d(final_pos[:, 0], final_pos[:, 1], bins=100)
plt.show()

