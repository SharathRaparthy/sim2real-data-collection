import numpy as np
from gym_ergojr.sim.abstract_robot import PusherRobot
import pybullet as p


class PusherRobotNoisy(PusherRobot):
    def __init__(self, action_noise=False, obs_noise=False):
        super(PusherRobotNoisy, self).__init__( debug=False)
        self.robot = None
        self.action_noise = action_noise
        self.obs_noise = obs_noise
        self.noise = 0.05
        super(PusherRobotNoisy, self).hard_reset()

    def act(self, action):
        action = action + np.random.uniform(-self.noise, + self.noise, 3)
        super(PusherRobotNoisy, self).act(action)

    def observe(self):
        obs = super(PusherRobotNoisy, self).observe()
        return obs + np.random.uniform(-self.noise, self.noise, 6)

    def get_tip(self):
        tip = 6  # TODO: might wanna check this in multi-robot setups
        state = np.asarray(p.getLinkState(self.robot, tip)[0])[:2]
        # pos = state[0]
        return state
