import pickle
import os
import matplotlib.pyplot as plt
# from nas.data import DATA_PATH
import numpy as np
from arguments import get_args
from arguments import get_args


args = get_args()
file_path = os.getcwd() + f'/data/freq{args.freq}/goal-babbling/'

pos_action_noise = np.load(file_path + f'goals_and_positions_freq-{args.freq}.npz')
fig, (ax1, ax2) = plt.subplots(1, 2)
position = pos_action_noise['positions']
print(position.shape)

ax1.hist2d(position[:, 0], position[:, 1], bins=100)
file_path = os.getcwd() + f'/data/freq{args.freq}/motor-babbling/'
ax1.set_title('Goal Babbling')
pos_action_noise = np.load(file_path + f'random_end_pos_{args.freq}.npy')
position = pos_action_noise
ax2.hist2d(position[:, 0], position[:, 1], bins=100)
ax2.set_title('Motor Babbling')
fig.suptitle(f"2D Histogram of end effector positions | Freq - {args.freq}")
plt.savefig(f'freq-{args.freq}.png', figsize=(20, 10))
plt.show()