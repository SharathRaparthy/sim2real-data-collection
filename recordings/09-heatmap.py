import pickle
import os
import matplotlib.pyplot as plt
# from nas.data import DATA_PATH
import numpy as np
from arguments import get_args
from arguments import get_args


args = get_args()
file_path = os.getcwd() + '/data/ErgoPusher/freq{}/goal-babbling/'.format(args.freq)

pos_action_noise = np.load(file_path + 'goals_and_positions.npz')
fig, (ax1, ax2) = plt.subplots(1, 2)
goals = pos_action_noise['goals']
position = pos_action_noise['positions']
print(position[:300])

ax1.hist2d(position[:, 0], position[:, 1], bins=100)
file_path = os.getcwd() + '/data/{}/freq{}/motor-babbling/'.format(args.env_name, args.freq)
ax1.set_title('Goal Babbling')
end_pos = np.load(file_path + 'end_positions.npy'.format(args.freq))
position = end_pos
print(position)
ax2.hist2d(position[:, 1], position[:, 2], bins=100)
ax2.set_title('Motor Babbling')
fig.suptitle("2D Histogram of end effector positions | Freq - {}".format(args.freq))
plt.savefig('freq-{}.png'.format(args.freq), figsize=(20, 10))
plt.show()
