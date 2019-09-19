import numpy as np
import os
from gym_ergojr.sim.single_robot import SingleRobot
import pickle
from arguments import get_args


args = get_args()

file_path = os.getcwd() + f'/data/freq{args.freq}/{args.approach}/'

actions, trajectories = np.load(file_path + 'actions_trajectories_10000.npz')["actions"], np.load(file_path + 'real_world_trajectories.npy')


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
    obs = robot.observe()
    # execute the action on simulator
    if epi % args.freq == 0:
        action = actions[epi, :]
    else:
        action += np.random.normal(0, 0.01)
        action[0], action[3] = 0, 0
    robot.act2(action)
    robot.step()
    obs = robot.observe()
    dataset["next-real-posvel"][epi, :] = trajectories[epi + 1, :]
    dataset["next-sim-posvel"][epi, :] = obs

f = open(file_path + f'99-lstm-{args.approach}-data.pkl', 'wb')
pickle.dump(dataset, f)
f.close()



