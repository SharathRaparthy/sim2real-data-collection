import time
import random
import os
import numpy as np
from scripts.goal_babbling import GoalBabbling
import matplotlib.pyplot as plt
from arguments import get_args
from gym_ergojr.sim.objects import Puck
from common.envs import NewPuck

args = get_args()

random.seed(args.seed)
np.random.seed(seed=args.seed)

total_steps = args.total_steps
rest_interval = args.rest_interval
freq = args.freq
steps_until_resample = args.num_steps/freq

# Hyper-parameters
SAMPLE_NEW_GOAL = args.goal_sample_freq
NUMBER_OF_RETRIES = args.num_retries
ACTION_NOISE = 0.4
K_NEAREST_NEIGHBOURS = 8
EPSILON = 0.3
task = args.task
goal_babbling = GoalBabbling(ACTION_NOISE, NUMBER_OF_RETRIES, task)
puck = NewPuck()
# Reset the robot
goal_babbling.reset_robot()

end_pos = []
history = []
max_history_len = args.history_len
goal_positions = []
count = 0

file_path = '/home/sharath/sim2real-data-recordings/'
if not os.path.isdir(file_path + 'data/ErgoPusher/freq{}/{}'.format(args.freq, args.approach)):
    os.makedirs(file_path + 'data/ErgoPusher/freq{}/{}'.format(args.freq, args.approach))


print('================================================')
print('Approach is : {} | Task is {} | Frequency is : {}'.format(args.approach, task, args.freq))
print('================================================')


# Create numpy arrays to store actions and observations
sim_trajectories = np.zeros((total_steps, 8))
actions = np.zeros((total_steps, goal_babbling.action_len))
for epi in range(total_steps):
    if epi % rest_interval == 0:  # Reset the robot after every rest interval
        print('Taking Rest at {}'.format(epi))
        goal_babbling.reset_robot()

    if epi % steps_until_resample == 0:
        # goal = [random.uniform(-0.1436, 0.22358), random.uniform(0.016000, 0.25002)]  # Reacher goals
        # goal = [random.uniform(-0.135, 0.0), random.uniform(-0.081, 0.135)]  # Pusher goals
        puck.hard_reset()
        goal = puck.puck_pos() # goals are not corelating with tip positions. Take care of normalization
        if count < 10:
            action = goal_babbling.sample_action()
        else:
            action = goal_babbling.sample_action() if random.random() < EPSILON \
                else goal_babbling.action_retries(goal, history)
        count += 1
    else:
        if task == 'reacher':
            action[0], action[3] = 0, 0
    _, end_position, observation = goal_babbling.perform_action(action)  # Perform the action and get the observation
    if len(history) >= max_history_len:
        del history[0]
    history.append((action, end_position))  # Store the actions and end positions in buffer
    end_pos.append(end_position)
    goal_positions.append(goal)
    actions[epi, :] = action  # Store the actions
    sim_trajectories[epi, :] = np.hstack([observation, puck.normalize_puck()])  # Store the observations

# Save the end positions, goals, actions and simulation trajectories.
final_pos = np.asarray(end_pos)
final_goals = np.asarray(goal_positions)
np.savez(file_path + 'data/ErgoPusher/freq{}/{}/goals_and_positions.npz'.format(args.freq, args.approach)
         , positions=final_pos, goals=final_goals)
np.savez(file_path + 'data/ErgoPusher/freq{}/{}/actions_trajectories.npz'.format(args.freq, args.approach),
         actions=actions, sim_trajectories=sim_trajectories)

# Plot the end_pos, goals and 2D histogram of end_pos
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12, 6))
ax1.scatter(final_pos[:, 0], final_pos[:, 1], alpha=0.5, linewidths=1)
ax1.set_xlim(-0.135, 0.0) # Change axis limits | Pusher : x(-0.135, 0.0) | Reacher : -0.1436, 0.22358
ax1.set_ylim(-0.081, 0.135) # Change axis limits | Pusher : x(-0.081, 0.135) | Reacher (0.016000, 0.25002)
ax1.set_title("End effector positions for {} trajectories".format(total_steps / 100))
ax2.scatter(final_goals[:, 0], final_goals[:, 1], alpha=0.5, linewidths=1)
ax2.set_title('Goals sampled')
plt.savefig(file_path + 'data/ErgoPusher/freq{}/{}/positions-goals.png'.format(freq, args.approach))
plt.close()
# Plot the 2D histogram and save it.
plt.hist2d(final_pos[:, 0], final_pos[:, 1], bins=100)
plt.xlim(-0.135, 0.0)
plt.ylim(-0.081, 0.135)
plt.title("2D Histogram of end effector positions")
plt.savefig(file_path + 'data/ErgoPusher/freq{}/{}/histogram.png'.format(freq, args.approach))

