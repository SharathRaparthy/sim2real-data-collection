from dm_control import suite
from dm_control import viewer
import numpy as np


env = suite.load(domain_name="finger", task_name="turn_easy")

action_spec = env.action_spec()
time_step = env.reset()

def random_policy(time_step):
    while not time_step.last():
      action = np.random.uniform(action_spec.minimum,
                                 action_spec.maximum,
                                 size=action_spec.shape)
      time_step = env.step(action)
      print(time_step.reward, time_step.discount, time_step.observation)

viewer.launch(env, policy=random_policy)
