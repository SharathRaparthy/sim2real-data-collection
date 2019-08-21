import numpy as np
import gym
import time
import random
from gym_ergojr.sim.single_robot import SingleRobot
import learners

class GoalBabbling(object):
    def __init__(self, action_noise, num_retires):
        self._nn_set = learners.NNSet()
        self.noise = action_noise
        self.num_retires = num_retires
        self.goal_y_range = [-0.12, 0.12]
        self.goal_z_range = [0.02, 0.24]

    @staticmethod
    def dist(a, b):
        return sum((a_i-b_i)**2 for a_i, b_i in zip(a, b))

    def nearest_neighbor(self, goal, history):
        """Return the motor command of the nearest neighbor of the goal"""
        nn_command, nn_dist = None, float('inf')  # naive nearest neighbor search.
        for m_command, effect in history:
            if self.dist(effect, goal) < nn_dist:
                nn_command, nn_dist = m_command, self.dist(effect, goal)
        return nn_command

    def inverse(self, goals, history):
        """Transform a goal into a motor command"""
        nn_command = self.nearest_neighbor(goals, history)  # find the nearest neighbor of the goal.
        new_command = []
        for m_i in nn_command:
            max_i = min(1, m_i + 2 * self.noise * 1)
            min_i = max(-1, m_i + 2 * self.noise * -1)
            new_command.append(random.uniform(min_i, max_i))
        return new_command



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
