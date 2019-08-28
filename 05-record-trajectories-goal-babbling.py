import random
import os
import numpy as np
from gym_ergojr.sim.single_robot import SingleRobot
from scripts.goal_babbling import GoalBabbling
import matplotlib.pyplot as plt

seed=225
random.seed(seed)
np.random.seed(seed=seed)
method = "goal-babbling"

total_steps = 2 * 100
rest_interval = 10 * 100
freq = 10
count = 0
steps_until_resample = 100/freq

# Hyperparameters
SAMPLE_NEW_GOAL = 1
NUMBER_OF_RETRIES = 5
ACTION_NOISE = 0.2
K_NEAREST_NEIGHBOURS = 8
EPSILON = 0.2

goal_babbling = GoalBabbling(ACTION_NOISE, NUMBER_OF_RETRIES)

# Reset the robot
robot = SingleRobot(debug=False)  
robot.reset()
robot.step()

end_pos = []
history = []
max_history_len = 10000
goal_positions = []
count = 0

file_path = '/home/sharath/sim2real-data-collection/'
if not os.path.isdir(file_path + f'data/freq{freq}/{method}'):
    os.makedirs(file_path + f'data/freq{freq}/{method}')
    print('yeyy')

# Create numpy arrays to store actions and observations
sim_trajectories = np.zeros((total_steps, 12))
actions = np.zeros((total_steps, 6))
for epi in range(total_steps):
    if epi % rest_interval == 0:  # Reset the robot after every rest interval
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
    _, end_position, observation = goal_babbling.perform_action(action)  # Perform the action and get the observation
    if len(history) >= max_history_len:
        del history[0]
    history.append((action, end_position))  # Store the actions and end positions in buffer
    end_pos.append(end_position)
    goal_positions.append(goal)
    actions[epi, :] = action  # Store the actions
    sim_trajectories[epi, :] = observation  # Store the observations


# Save the end positions, goals, actions and simulation trajectories.
final_pos = np.asarray(end_pos)
final_goals = np.asarray(goal_positions)
np.savez(file_path + f'data/freq{freq}/{method}/goals_and_positions.npz', positions=final_pos, goals=final_goals)
np.savez(file_path + f'data/freq{freq}/{method}/actions_trajectories_{total_steps / 100}.npz', actions=actions, sim_trajectories=sim_trajectories)

# Plot the end_pos, goals and 2D histogram of end_pos
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
ax1.scatter(final_pos[:, 0], final_pos[:, 1], alpha=0.5, linewidths=1)
ax1.set_xlim(-0.1436, 0.22358)
ax1.set_ylim(0.016000, 0.25002)
ax1.set_title(f"End effector positions for {total_steps / 100} trajectories")
ax2.scatter(final_goals[:, 0], final_goals[:, 1], alpha=0.5, linewidths=1)
ax2.set_title(f'Goals sampled')
plt.savefig(file_path + f'data/freq{freq}/{method}/positions-goals.png')
plt.close()
# Plot the 2D histogram and save it.
plt.hist2d(final_pos[:, 0], final_pos[:, 1], bins=100)
plt.xlim(-0.1436, 0.22358)
plt.ylim(0.016000, 0.25002)
plt.title("2D Histogram of end effector positions")
plt.savefig(file_path + f'data/freq{freq}/{method}/histogram.png')

