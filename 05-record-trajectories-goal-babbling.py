import numpy as np
import random
from gym_ergojr.sim.single_robot import SingleRobot
from scripts.goal_babbling import GoalBabbling
import matplotlib.pyplot as plt

total_steps = 400 * 100
rest_interval = 10 * 100
freq = 1
count = 0
steps_until_resample = 100/freq

#HYPERPARAMETERS
SAMPLE_NEW_GOAL = 1
NUMBER_OF_RETRIES = 2
ACTION_NOISE = 0.1
K_NEAREST_NEIGHBOURS = 8
EPSILON = 0.2

goal_babbling = GoalBabbling(ACTION_NOISE, NUMBER_OF_RETRIES)

# Reset the robot
robot = SingleRobot(debug=False)
robot.reset()
robot.step()

end_pos = []
history = []

# Perform some initial random action and append to the history
action, end_position = goal_babbling.sample_action()
history.append((action, end_position))

for epi in range(total_steps):
    history_local = []
    if epi % SAMPLE_NEW_GOAL == 0:
        goal = [random.uniform(-0.12, 0.12), random.uniform(0.02, 0.24)]

    if random.random() < EPSILON:
        action, end_position = goal_babbling.sample_action()
    else:
        action, end_position = goal_babbling.action_retries(goal, history)

    history.append((action, end_position))
    end_pos.append(end_position)

final_pos = np.asarray(end_pos)
plt.scatter(final_pos[:, 0], final_pos[:, 1], alpha=0.5)
plt.show()
plt.hist2d(final_pos[:, 0], final_pos[:, 1], bins=100)
plt.show()

