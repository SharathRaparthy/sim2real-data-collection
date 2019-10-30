import random
import numpy as np
from gym_ergojr.sim.single_robot import SingleRobot
from gym_ergojr.sim.abstract_robot import PusherRobot
import learners


class GoalBabbling(object):
    def __init__(self, action_noise, num_retries, task):
        self.noise = action_noise
        self.retries = num_retries
        self.task = task
        if task == 'pusher':
            self.robot = PusherRobot()
        else:
            self.robot = SingleRobot(debug=False)
        self.action_len = len(self.robot.motor_ids)
        self._nn_set = learners.NNSet()
        np.random.seed(seed=225)
        random.seed(225)

    def nearest_neighbor(self, goal, history):
        """Return the motor command of the nearest neighbor of the goal"""
        if len(history) < len(self._nn_set):  # HACK
            self._nn_set = learners.NNSet()
        for m_command, effect in history[len(self._nn_set):]:
            self._nn_set.add(m_command, y=effect)
        idx = self._nn_set.nn_y(goal)[1][0]
        return history[idx][0]

    def add_noise(self, nn_command):
        action = np.asarray(nn_command)
        action_noise = np.random.uniform(-self.noise, self.noise, self.action_len)
        new_command = action + action_noise
        if self.task == 'reacher':
            new_command[0], new_command[3] = 0, 0
        new_command = np.clip(new_command, -1, 1)
        return new_command

    def sample_action(self):
        action = np.random.uniform(-1, 1, self.action_len)
        if self.task == 'reacher':
            action[0], action[3] = 0, 0
        return action

    def action_retries(self, goal, history):
        history_local = []
        goal = np.asarray([goal])
        action = self.nearest_neighbor(goal, history)
        for _ in range(self.retries):
            action_noise = self.add_noise(action)
            _, end_position, obs = self.perform_action(action_noise)
            history_local.append((action_noise, end_position))
        action_new = self.nearest_neighbor(goal, history_local)
        return action_new

    def perform_action(self, action):
        self.robot.act(action)
        self.robot.step()
        end_pos = self.robot.get_tip()[1:]
        obs = self.robot.observe()
        return action, end_pos, obs

    def reset_robot(self):
        if self.task == 'pusher':
            self.robot.hard_reset()
        self.robot.rest()
        self.robot.step()

    @staticmethod
    def dist(a, b):
        return np.linalg.norm(a-b) # We dont need this method anymore


if __name__ == '__main__':
    goal_babbling = GoalBabbling()
    history_test = []
    for i in range(1000):  # comparing the results over 1000 random query.
        m_command = [random.random() for _ in range(4)]
        effect = [random.random() for _ in range(2)]
        history_test.append((m_command, effect))

        goal = [random.random() for _ in range(2)]
        nn_a = goal_babbling.random_goal_babbling(history_test)
        print(nn_a)
