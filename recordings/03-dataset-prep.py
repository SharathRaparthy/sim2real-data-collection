import numpy as np
import os
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.sim.abstract_robot import PusherRobot
import pickle
from arguments import get_args
from common.envs import NewPuck
from common.envs import PusherRobotNoisy

''' Prepare the dataset for training LSTM '''

args = get_args()

file_path = os.getcwd() + '/data/{}/freq{}/{}/'.format(args.env_name, args.freq, args.approach)
robot = PusherRobotNoisy(action_noise=args.action_noise, obs_noise=args.obs_noise) # 3DOF Pusher

actions, trajectories = np.load(file_path + 'actions_trajectories.npz')["actions"],\
                        np.load(file_path + 'real_world_trajectories_{}.npy'.format(args.noise_type))


dataset = {
    "real-posvel": np.zeros(trajectories.shape),
    "actions": np.zeros(actions.shape),
    "next-real-posvel": np.zeros(trajectories.shape),
    "next-sim-posvel": np.zeros(trajectories.shape),
}
puck = NewPuck()
# trajectories[:, :6] += 90
# trajectories[:, :6] /= 180
# trajectories[:, :6] = trajectories[:, :6]*2 - 1
# trajectories[:, 6:] += 150
# trajectories[:, 6:] /= 300
# trajectories[:, 6:] = trajectories[:, 6:]*2 - 1
print(trajectories[:200, :])
print(actions[:200, :])
robot.hard_reset()
puck.hard_reset()
robot.rest()
robot.step()
for epi in range(actions.shape[0]):
    if epi + 1 >= actions.shape[0]:
        continue

    # collect the data and store in dictionary
    dataset["actions"][epi, :] = actions[epi, :]
    dataset["real-posvel"][epi, :] = trajectories[epi, :]
    # rest the position to real-posvel
    robot.set(dataset["real-posvel"][epi, :6])
    robot.step()
    robot_obs = robot.observe()
    puck_obs = puck.normalize_puck()
    obs = np.hstack([robot_obs, puck_obs])

    # execute the action on simulator
    if epi % args.freq == 0:
        action = actions[epi, :]

    robot.act(action)
    robot.step()
    robot_obs = robot.observe()
    puck_obs = puck.normalize_puck()
    obs = np.hstack([robot_obs, puck_obs])
    dataset["next-real-posvel"][epi, :] = trajectories[epi + 1, :]
    dataset["next-sim-posvel"][epi, :] = obs

f = open(file_path + '{}-lstm-{}-data-{}.pkl'.format(args.freq, args.approach, args.noise_type), 'wb')
pickle.dump(dataset, f)
f.close()



