import matplotlib.pyplot as plt
import numpy as np
import os
file_path = os.getcwd() + '/files/225/numpy_files/pos-action-noise-0.2-retries-5-eps-0.2.npz'
file = np.load(file_path)
final_pos = file["position"]
final_goals = file["goals"]
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(12,6))
ax1.scatter(final_pos[:, 0], final_pos[:, 1], alpha=0.5, linewidths=1)
ax1.set_xlim(-0.1436, 0.22358)
ax1.set_ylim(0.016000, 0.25002)
ax1.set_title(f"End effector positions")
ax2.scatter(final_goals[:, 0], final_goals[:, 1], alpha=0.5, linewidths=1)
ax2.set_title(f'Goals sampled')
plt.show()