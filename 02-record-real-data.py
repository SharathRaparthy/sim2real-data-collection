import time
import numpy as np
from poppy_helpers.controller import ZMQController

zmq = ZMQController('flogo3.local')
zmq.compliant(False)
zmq.set_max_speed(100)

file_path = '/home/sharath/sim2real-record/data/freq10/'

rest_interval = 10 * 100
freq = 10

steps_until_resample = 100/freq

actions_trajectories = np.load(file_path + '04-clean_action_trajectories.npz')
actions = actions_trajectories["actions"]
real_trajectories = np.zeros((actions.shape[0], 12))

for epi in range(actions.shape[0]):
    start = time.time()

    if epi % rest_interval == 0:
        print(f"Episodes completed: {epi}")
        zmq.rest()
        time.sleep(0.25)
    action = actions[epi, :]
    zmq.goto_normalized(action)

    real_trajectories[epi, :] = zmq.get_posvel()

    delta = start - time.time()

    if delta < 0.01:
        time.sleep(0.01 - delta)

    if epi % 1000 == 0:
        np.save(file_path + '04-real_world_trajectories.npy', real_trajectories)


np.save(file_path + '04-real_world_trajectories.npy', real_trajectories)








