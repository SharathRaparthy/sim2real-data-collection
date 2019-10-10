import time

from gym_ergojr.sim.single_robot import SingleRobot



robot = SingleRobot(debug=True)
robot.reset()
robot.step()

for i in range(100):
    # robot.act2([0,-1,0,0,-1,0]) # min y: -0.1436
    # robot.act2([0,1,-1,0,0,0]) # max y: 0.22358
    # robot.act2([0,-.2,0,0,-.8,0]) # max z: 0.25002
    robot.act2([0,-.2,1,0,0,0.5]) # min z: 0.016000
    robot.step()
    print (robot.get_tip()[0][1:]) # cut off x axis, only return y (new x) and z (new y)
    time.sleep(0.1)
