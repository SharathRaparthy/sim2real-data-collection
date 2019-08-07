import numpy as np
import os
from gym_ergojr.sim.single_robot import SingleRobot
import pickle


file_path = os.getcwd() + '/data/freq10/'

action_1, trajectories_1 = np.load(file_path + '01-clean_action_trajectories.npz')["actions"], np.load(file_path + '01-real_world_trajectories.npy')
action_2, trajectories_2 = np.load(file_path + '02-clean_action_trajectories.npz')["actions"], np.load(file_path + '02-real_world_trajectories.npy')
action_3, trajectories_3 = np.load(file_path + '03-clean_action_trajectories.npz')["actions"], np.load(file_path + '03-real_world_trajectories.npy')
action_4, trajectories_4 = np.load(file_path + '04-clean_action_trajectories.npz')["actions"][:100000,:], np.load(file_path + '04-real_world_trajectories.npy')[:100000,:]
# actions = np.concatenate([action_1, action_2, action_3], axis=0)
# trajectories = np.concatenate([trajectories_1, trajectories_2, trajectories_3], axis=0)
actions = np.concatenate([action_1, action_2, action_3], axis=0)
trajectories = np.concatenate([trajectories_1, trajectories_2, trajectories_3], axis=0)

frequency = 1

robot = SingleRobot(debug=False)
robot.reset()
robot.step()
dataset = {
    "real-posvel": np.zeros(trajectories.shape),
    "actions": np.zeros(actions.shape),
    "next-real-posvel": np.zeros(trajectories.shape),
    "next-sim-posvel": np.zeros(trajectories.shape),
}

trajectories[:, :6] += 90
trajectories[:, :6] /= 180
trajectories[:, :6] = trajectories[:, :6]*2 - 1
trajectories[:, 6:] += 150
trajectories[:, 6:] /= 300
trajectories[:, 6:] = trajectories[:, 6:]*2 - 1
print(trajectories[1, :])
print(actions.shape)
robot.reset()
robot.step()
for epi in range(actions.shape[0]):
    if epi + 1 >= actions.shape[0]:
        continue
    # collect the data and store in dictionary
    dataset["actions"][epi, :] = actions[epi, :]
    dataset["real-posvel"][epi, :] = trajectories[epi, :]
    # reset the position to real-posvel
    robot.set(dataset["real-posvel"][epi, :])
    robot.step()
    # execute the action on simulator
    robot.act2(actions[epi, :])
    robot.step()
    obs = robot.observe()
    dataset["next-real-posvel"][epi, :] = trajectories[epi + 1, :]
    dataset["next-sim-posvel"][epi, :] = obs



f = open(file_path + '10-lstm-data.pkl', 'wb')
pickle.dump(dataset, f)
f.close()



