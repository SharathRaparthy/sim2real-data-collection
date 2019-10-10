import pickle
import os
import matplotlib.pyplot as plt
# from nas.data import DATA_PATH
import numpy as np
from arguments import get_args
from arguments import get_args


args = get_args()
file_path = os.getcwd() + f'/data/freq{args.freq}/{args.approach}/'

pos_action_noise = np.load(file_path + 'goals_and_positions.npz')

position = pos_action_noise['positions']
print(position.shape)

plt.hist2d(position[:, 0], position[:, 1], bins=100)

plt.title("2D Histogram of end effector positions")
plt.show()