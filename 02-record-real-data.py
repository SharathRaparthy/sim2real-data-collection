import time
import os
import numpy as np
from poppy_helpers.controller import ZMQController
from arguments import get_args

args = get_args()

zmq = ZMQController('flogo3.local')
zmq.compliant(False)
zmq.set_max_speed(100)

file_path = os.getcwd() + f'/data/freq{args.freq}/{args.approach}/'

rest_interval = 10 * 100
freq = args.freq
steps_until_resample = 100/freq

actions_trajectories = np.load(file_path + 'action_trajectories.npz')
actions = actions_trajectories["actions"]
real_trajectories = np.zeros((actions.shape[0], 12))

for epi in range(actions.shape[0]):
    start = time.time()

    if epi % rest_interval == 0:
        print(f"Episodes completed: {epi}")
        zmq.rest()
    action = actions[epi, :]
    zmq.goto_normalized(action)

    real_trajectories[epi, :] = zmq.get_posvel()

    delta = start - time.time()

    if delta < 0.01:
        time.sleep(0.01 - delta)

    if epi % 1000 == 0:
        np.save(file_path + 'real_world_trajectories.npy', real_trajectories)


np.save(file_path + 'real_world_trajectories.npy', real_trajectories)








