"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
from pynput import keyboard
from pynput.keyboard import Key
import numpy as np
import gym
import gym_ergojr
from sim.pusher_robot import PusherRobotNoisy
from sim.objects import NewPuck

# env = gym.make('ErgoPusher-Graphical-v1')
robot = PusherRobotNoisy()
puck = NewPuck()
puck.hard_reset()
ACTIONS = [
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, -1.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0])
]

ACTION_KEYS = [Key.up, Key.down, Key.page_up , Key.page_down, Key.right, Key.left]



def on_press(key):
    global reward
    if key in ACTION_KEYS:
        # s, r, d, info = env.step(ACTIONS[ACTION_KEYS.index(key)])
        robot.act(ACTIONS[ACTION_KEYS.index(key)])
        robot.step()
        tip_pos = robot.get_tip()
        goal = puck.puck_pos()
        # env.render()
        # reward += r
        # pos = env.get_tip_position()

        print('Tip Position: {}'.format(np.around(tip_pos, 3)))
        # print('Goal Pos : {}'.format(goal))
        print('Puck Pos : {}'.format(goal))
        # if d:
            # env.reset()
            # reward = 0

with keyboard.Listener(on_press=on_press) as listener:
    listener.join()

env.close()
