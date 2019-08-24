import random
import numpy as np
from gym_ergojr.sim.single_robot import SingleRobot
import learners


class GoalBabbling(object):
    def __init__(self, action_noise, num_retries):
        self.noise = action_noise
        self.retries = num_retries
        self.goal_y_range = [-0.12, 0.12]
        self.goal_z_range = [0.02, 0.24]
        self.robot = SingleRobot(debug=False)
        self._nn_set = learners.NNSet()

    def nearest_neighbor(self, goal, history):
        """Return the motor command of the nearest neighbor of the goal"""
        if len(history) < len(self._nn_set):  # HACK
            self._nn_set = learners.NNSet()
        for m_command, effect in history[len(self._nn_set):]:
            self._nn_set.add(m_command, y=effect)
        idx = self._nn_set.nn_y(goal)[1][0]
        return history[idx][0]

    def add_noise(self, nn_command):
        new_command = []
        for m_i in nn_command:
            max_i = min(1, m_i + 2 * self.noise * 1)
            min_i = max(-1, m_i + 2 * self.noise * -1)
            new_command.append(random.uniform(min_i, max_i))
        new_command[0], new_command[3] = 0, 0
        return new_command

    def sample_action(self):
        action = np.random.uniform(-1, 1, 6)
        action[0], action[3] = 0, 0
        return action

    def action_retries(self, goal, history):
        history_local = []
        goal = np.asarray([goal])
        action = self.nearest_neighbor(goal, history)
        for _ in range(self.retries):
            action_noise = self.add_noise(action)
            _, end_position = self.perform_action(action_noise)
            history_local.append((action_noise, end_position))
        action_new = self.nearest_neighbor(goal, history_local)
        return action_new

    def perform_action(self, action):
        self.robot.act2(action)
        self.robot.step()
        end_pos = self.robot.get_tip()[0][1:]
        return action, end_pos

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
