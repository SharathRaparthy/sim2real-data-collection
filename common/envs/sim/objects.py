import numpy as np
from gym_ergojr.sim.objects import Puck
import pybullet as p

class NewPuck(Puck):
    def __init__(self):
        super(NewPuck, self).__init__()
        self.puck = None
        self.dbo = None
        self.target = None
        self.goal = None
        self.obj_visual = None

    def add_puck(self):
        super(NewPuck, self).add_puck()

    def puck_pos(self):
        tip = 1
        pos = np.array(p.getLinkState(self.puck, tip)[0])[:2]
        return pos


# ------------------- Pusher env
#     def _get_state(self):
#         return self.robot.observe()
#
#     def get_tip_position(self):
#         return self.robot.get_tip()
#     def _set_state_(self, state):
#         self.robot.set(state)
