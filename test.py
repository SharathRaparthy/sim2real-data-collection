import time
import numpy as np
from poppy_helpers.controller import ZMQController

zmq = ZMQController('flogo3.local')
zmq.compliant(False)
zmq.set_max_speed(100)

action = np.array([0, 1, 0, 0, 0,0])
zmq.goto_normalized(action)
zmq.rest()