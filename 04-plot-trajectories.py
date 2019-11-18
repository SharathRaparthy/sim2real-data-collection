import pickle
import os
import matplotlib.pyplot as plt
# from nas.data import DATA_PATH
import numpy as np
from arguments import get_args


args = get_args()
file_path = os.getcwd() + '/data/{}/freq{}/goal-babbling/'.format(args.env_name, args.variant)


data = pickle.load(open(os.path.join(file_path, "{}-lstm-goal-babbling-data.pkl".format(args.variant)), "rb"))

print(data.keys())  # ['real-posvel', 'actions', 'next-real-posvel', 'next-sim-posvel']
real_posvel = data['real-posvel']
actions = data["actions"]
next_real_posvel = data["next-real-posvel"]
next_sim_posvel = data["next-sim-posvel"]

print(real_posvel.shape)
print(actions.shape)
print(next_real_posvel.shape)
print(next_sim_posvel.shape)

real_shape = int(real_posvel.shape[1] / 2)
print(real_shape)
print("real_pos", real_posvel[:, real_shape].min(), real_posvel[:, real_shape].max(), real_posvel[:, real_shape].mean())
print("real_vel", real_posvel[:, real_shape:].min(), real_posvel[:, real_shape:].max(), real_posvel[:, real_shape:].mean())
print("actions", actions.min(), actions.max(), actions.mean())
print("next_real_pos", next_real_posvel[:, :real_shape].min(), next_real_posvel[:, :real_shape].max(), next_real_posvel[:, :real_shape].mean())
print("next_real_vel", next_real_posvel[:, real_shape:].min(), next_real_posvel[:, real_shape:].max(), next_real_posvel[:, real_shape:].mean())
print("next_sim_pos", next_sim_posvel[:, :real_shape].min(), next_sim_posvel[:, :real_shape].max(), next_sim_posvel[:, :real_shape].mean())
print("next_sim_vel", next_sim_posvel[:, real_shape:].min(), next_sim_posvel[:, real_shape:].max(), next_sim_posvel[:, real_shape:].mean())

start = 0
samples = 100
end = start + samples

x = np.arange(start, end)
#
#
# print ("first motor REAL:",real_posvel[:,0].min(), real_posvel[:,0].max(), real_posvel[:,0].mean())
# print ("first motor SIM:",next_sim_posvel[:,0].min(), next_sim_posvel[:,0].max(), next_sim_posvel[:,0].mean())
#
# for i in range(6):
#     print (f"actions, motor{i}:", actions[:,i].min(), actions[:,i].max(), actions[:,i].mean())

# ==================== REAL
print(real_posvel[:, :6])

for i in range(2):
    plt.plot(x, next_real_posvel[start:end, i], label="motor {} next sim pos".format(i))
    plt.plot(x, next_sim_posvel[start:end, i], label="motor {} next real pos".format(i))
    # plt.plot(x, real_posvel[start:end, i + real_shape], label="motor {} vel".format(i), linestyle="dashed")
    plt.plot(x, actions[start:end, i], label="motor {} action - V{}".format(i, args.variant), linestyle="dotted")

plt.title("100Hz REAL goal babbling recordings from timestep {} to {}".format(start, end))
plt.legend()
plt.tight_layout()

plt.show()

# ==================== SIM

file_path = os.getcwd() + '/data/ErgoPusher/freq{}/goal-babbling/'.format(args.variant)
data = pickle.load(open(os.path.join(file_path, "{}-lstm-goal-babbling-data-action_noise.pkl".format(args.variant)), "rb"))

real_posvel = data['real-posvel']
actions = data["actions"]
next_real_posvel = data["next-real-posvel"]
next_sim_posvel = data["next-sim-posvel"]
for i in range(2):
    plt.plot(x, next_sim_posvel[start:end, i], label="motor {} pos".format(i))
    plt.plot(x, next_sim_posvel[start:end, i + real_shape], label="motor {} vel".format(i), linestyle="dashed")
    plt.plot(x, actions[start:end, i ], label="motor {} action - V{}".format(i, args.variant), linestyle="dotted")
plt.title("100Hz SIM goal babbling recordings from timestep {} to {}".format(start, end))
plt.legend()
plt.tight_layout()

plt.show()
