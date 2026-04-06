import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

from utils import camFormat

def plot_frame(ax, R, label, color=['r', 'g', 'b']):
    """Plot a coordinate frame given a rotation matrix."""
    origin = np.array([0, 0, 0])
    x_axis, y_axis, z_axis = R @ np.eye(3)  # Transform standard basis vectors

    ax.quiver(*origin, *x_axis, color=color[0], label=f'{label} X', length=1.0, linewidth=2)
    ax.quiver(*origin, *y_axis, color=color[1], label=f'{label} Y', length=1.0, linewidth=2)
    ax.quiver(*origin, *z_axis, color=color[2], label=f'{label} Z', length=1.0, linewidth=2)

# Define the rotation matrices
R_tilt = R.from_euler('x', 5, degrees=True).as_matrix()
R_yhoriz = R.from_euler('y', 70, degrees=True).as_matrix()
R_yup = R.from_euler('x', 90, degrees=True).as_matrix()
R_view = R.from_euler('z', 90, degrees=True).as_matrix()

# Compute total rotation
R_total = R_view @ R_yup @ R_yhoriz @ R_tilt

R_total_calc_inv = R_tilt @ R_yhoriz @ R_yup @ R_view

# exacly how I do it
rot_tilt_x = 5
rot_horizontal_y = 70
rotation_angles = [rot_tilt_x, rot_horizontal_y]
transMatrix = camFormat.getCamTransform_np([0,0,0], rotation_angles, rotation_axis='xy')
rot_my = transMatrix[:3, :3]
rot_my = camFormat.init_to_opengl(rot_my)

my_rot_w2c = np.linalg.inv(rot_my)

# Plot step-by-step transformations
fig = plt.figure(figsize=(11, 11))
axes = [fig.add_subplot(221, projection='3d'),
        fig.add_subplot(222, projection='3d'),
        fig.add_subplot(223, projection='3d'),
        fig.add_subplot(224, projection='3d')]

# Initial Frame
plot_frame(axes[0], np.eye(3), 'Initial')
axes[0].set_title("Initial Camera Frame")

# After R_tilt
plot_frame(axes[1], R_tilt, 'Rtilt')
axes[1].set_title("After Tilt (5° around X)")

# After R_tilt @ R_yhoriz
plot_frame(axes[2], R_yhoriz @ R_tilt, 'Ryhoriz * Rtilt')
axes[2].set_title("After Horizon Rotation (100° around Y)")

# After R_yup @ R_yhoriz @ R_tilt
plot_frame(axes[3], R_yup @ R_yhoriz @ R_tilt, 'Ryup * Ryhoriz * Rtilt')
axes[3].set_title("After Up Rotation (90° around X)")

# Final Plot (Total Rotation)
fig_final = plt.figure(figsize=(4, 4))
ax_final = fig_final.add_subplot(111, projection='3d')
plot_frame(ax_final, R_total, 'Total')
ax_final.set_title("Final Rotation (Rview * Ryup * Ryhoriz * Rtilt)")

# Exact final rotation
fig_exact_final = plt.figure(figsize=(4, 4))
ax_exact_final = fig_exact_final.add_subplot(111, projection='3d')
plot_frame(ax_exact_final, rot_my, 'Exact Final')
ax_exact_final.set_title("My Rotation final")

# my w2c rotation
fig_my_w2c_final = plt.figure(figsize=(4, 4))
ax_my_w2c_final = fig_my_w2c_final.add_subplot(111, projection='3d')
plot_frame(ax_my_w2c_final, my_rot_w2c, 'My W2C')
ax_my_w2c_final.set_title("My Rotation final inverse (w2c)")

# calculated inverse rotation?
fig_calc_inv_final = plt.figure(figsize=(4, 4))
ax_calc_inv_final = fig_calc_inv_final.add_subplot(111, projection='3d')
plot_frame(ax_calc_inv_final, R_total_calc_inv, 'Calculated Inverse')
ax_calc_inv_final.set_title("Calculated Inverse Rotation")

# Formatting all plots
for ax in axes + [ax_final] + [ax_exact_final] + [ax_my_w2c_final] + [ax_calc_inv_final]:
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

plt.show()
