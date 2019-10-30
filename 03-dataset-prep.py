import numpy as np
import os
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.sim.abstract_robot import PusherRobot
import pickle
from arguments import get_args

''' Prepare the dataset for training LSTM '''

args = get_args()

file_path = os.getcwd() + '/data/{}/freq{}/{}/'.format(args.env_name, args.freq, args.approach)
robot = PusherRobot(action_noise=False, obs_noise=False, debug=False) # 3DOF Pusher

actions, trajectories = np.load(file_path + 'action_trajectories.npz')["actions"],\
                        np.load(file_path + 'real_world_trajectories_action_noise.npy')


dataset = {
    "real-posvel": np.zeros(trajectories.shape),
    "actions": np.zeros(actions.shape),
    "next-real-posvel": np.zeros(trajectories.shape),
    "next-sim-posvel": np.zeros(trajectories.shape),
}

# trajectories[:, :6] += 90
# trajectories[:, :6] /= 180
# trajectories[:, :6] = trajectories[:, :6]*2 - 1
# trajectories[:, 6:] += 150
# trajectories[:, 6:] /= 300
# trajectories[:, 6:] = trajectories[:, 6:]*2 - 1
print(trajectories[:200, :])
print(actions[:200, :])
robot.hard_reset()
robot.rest()
robot.step()
for epi in range(actions.shape[0]):
    if epi + 1 >= actions.shape[0]:
        continue

    # collect the data and store in dictionary
    dataset["actions"][epi, :] = actions[epi, :]
    dataset["real-posvel"][epi, :] = trajectories[epi, :]
    # rest the position to real-posvel
    robot.set(dataset["real-posvel"][epi, :])
    robot.step()
    obs = robot.observe()

    # execute the action on simulator
    if epi % args.freq == 0:
        action = actions[epi, :]

    robot.act(action)
    robot.step()
    obs = robot.observe()
    dataset["next-real-posvel"][epi, :] = trajectories[epi + 1, :]
    dataset["next-sim-posvel"][epi, :] = obs

# f = open(file_path + f'{args.freq}-lstm-{args.approach}-data-obs-noise.pkl', 'wb')
# pickle.dump(dataset, f)
# f.close()



