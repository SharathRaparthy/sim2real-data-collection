import pickle
import os
import matplotlib.pyplot as plt
# from nas.data import DATA_PATH
import numpy as np


file_path = os.getcwd() + '/data/freq1/goal-babbling/'

data = pickle.load(open(os.path.join(file_path, "01-lstm-goal-babbling-data.pkl"), "rb"))

print(data.keys())  # ['real-posvel', 'actions', 'next-real-posvel', 'next-sim-posvel']
real_posvel = data['real-posvel']
actions = data["actions"]
next_real_posvel = data["next-real-posvel"]
next_sim_posvel = data["next-sim-posvel"]

print(real_posvel.shape)
print(actions.shape)
print(next_real_posvel.shape)
print(next_sim_posvel.shape)

print("real_pos", real_posvel[:, 6].min(), real_posvel[:, 6].max(), real_posvel[:, 6].mean())
print("real_vel", real_posvel[:, 6:].min(), real_posvel[:, 6:].max(), real_posvel[:, 6:].mean())
print("actions", actions.min(), actions.max(), actions.mean())
print("next_real_pos", next_real_posvel[:, :6].min(), next_real_posvel[:, :6].max(), next_real_posvel[:, :6].mean())
print("next_real_vel", next_real_posvel[:, 6:].min(), next_real_posvel[:, 6:].max(), next_real_posvel[:, 6:].mean())
print("next_sim_pos", next_sim_posvel[:, :6].min(), next_sim_posvel[:, :6].max(), next_sim_posvel[:, :6].mean())
print("next_sim_vel", next_sim_posvel[:, 6:].min(), next_sim_posvel[:, 6:].max(), next_sim_posvel[:, 6:].mean())

start = 0
samples = 300
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

for i in range(3):
    plt.plot(x, real_posvel[start:end, i], label=f"motor {i} pos")
    plt.plot(x, real_posvel[start:end, i + 6], label=f"motor {i} vel", linestyle="dashed")
    plt.plot(x, actions[start:end, i], label=f"motor {i} action", linestyle="dotted")

plt.title(f"100Hz REAL goal babbling recordings from timestep {start} to {end}")
plt.legend()
plt.tight_layout()

plt.show()

# ==================== SIM


for i in range(3):
    plt.plot(x, next_sim_posvel[start:end, i], label=f"motor {i} pos")
    plt.plot(x, next_sim_posvel[start:end, i + 6], label=f"motor {i} vel", linestyle="dashed")
    plt.plot(x, actions[start:end, i], label=f"motor {i} action", linestyle="dotted")

plt.title(f"100Hz SIM goal babbling recordings from timestep {start} to {end}")
plt.legend()
plt.tight_layout()

plt.show()
