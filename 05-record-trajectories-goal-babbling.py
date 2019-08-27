import random
from collections import deque
import numpy as np
from gym_ergojr.sim.single_robot import SingleRobot
from scripts.goal_babbling import GoalBabbling
import matplotlib.pyplot as plt

seed=225
random.seed(seed)
np.random.seed(seed=seed)
total_steps = 200 * 100
rest_interval = 10 * 100
freq = 10
count = 0
steps_until_resample = 100/freq

#HYPERPARAMETERS
SAMPLE_NEW_GOAL = 1
NUMBER_OF_RETRIES = 5
ACTION_NOISE = 0.4
K_NEAREST_NEIGHBOURS = 8
EPSILON = 0.1

goal_babbling = GoalBabbling(ACTION_NOISE, NUMBER_OF_RETRIES)

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
            action = goal_babbling.sample_action() if random.random() < EPSILON \
                else goal_babbling.action_retries(goal, history)
        count += 1
    _, end_position = goal_babbling.perform_action(action)
    history.append((action, end_position))
    end_pos.append(end_position)
    goal_positions.append(goal)

final_pos = np.asarray(end_pos)
final_goals = np.asarray(goal_positions)
plt.scatter(final_pos[:, 0], final_pos[:, 1], alpha=0.5)
plt.xlim(-0.1436, 0.22358)
plt.ylim(0.016000, 0.25002)
plt.show()
plt.scatter(final_goals[:, 0], final_goals[:, 1], alpha=0.5)
plt.show()
plt.hist2d(final_pos[:, 0], final_pos[:, 1], bins=100)
plt.show()

