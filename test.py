from gym_ergojr.sim.single_robot import SingleRobot
import time
import numpy as np
robot = SingleRobot(debug=True)
action = [0, 0, -1, 0, -1, 1]
robot.reset()
robot.step()
posvel = np.zeros((12))
for _ in range(100):
    robot.act2(action)
    robot.step()
    obs = robot.observe()
    end_pos = robot.get_tip()[0][1:]
    print(f'End position before : {end_pos}')
    posvel[:6] = obs[:6]
    robot.set(posvel)
    robot.step()
    end_pos = robot.get_tip()[0][1:]
    print(f'End position after : {end_pos}')


