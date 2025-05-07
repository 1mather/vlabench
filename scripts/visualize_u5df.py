import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import os
from scipy.spatial.transform import Rotation as R
def quat2euler(quat):
    """convert quaternion to euler angle"""
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler = r.as_euler('xyz', degrees=False)  # radian
    return euler




def unwrap_angles(angles):
    """handle angle wrapping, make the angle change continuous"""
    return np.unwrap(angles, axis=0)

def plot_trajectory_comparison(filepath):
    # create output directory
    dataset_dir = Path(filepath).parent
    plot_dir = dataset_dir / "plot"
    plot_dir.mkdir(exist_ok=True)
    
    # load data
    with h5py.File(filepath, 'r') as f:
        data_group = list(f['data'].keys())[0]
        trajectory = np.array(f['data'][data_group]['trajectory'])
        ee_state = np.array(f['data'][data_group]['observation']['ee_state'])
        ee_euler = np.array([quat2euler(ee_state[i, 3:7]) for i in range(len(ee_state))])
        ee_euler = unwrap_angles(ee_euler)  # handle angle wrapping
    

    trajectory_angles = trajectory[:, 3:6]
    trajectory_angles = unwrap_angles(trajectory_angles)  # handle angle wrapping
    

    trajectory_unwrapped = np.concatenate([
        trajectory[:, :3],          # position
        trajectory_angles,          # processed angles
        trajectory[:, -1:],         # gripper
    ], axis=1)
    
    # combine new ee_state
    ee_state_euler = np.concatenate([
        ee_state[:, :3],  # position
        ee_euler,         # processed euler angle
        ee_state[:, -1:]  # gripper
    ], axis=1)
    
    # ee_state: [x, y, z, qx, qy, qz, qw, gripper]
    # trajectory: [x, y, z, rx, ry, rz, gripper]
    
    n_timesteps = trajectory.shape[0]
    time_steps = np.arange(n_timesteps)
    
    # create 8 subplots (8 dimensions of ee_state)
    fig, axes = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle('EE State and Trajectory Over Time')
    
    # 定义每个维度的标签
    dimensions = [
        (0, "X Position", "Position (m)"),
        (1, "Y Position", "Position (m)"),
        (2, "Z Position", "Position (m)"),
        (3, "Roll", "Value"),
        (4, "Pitch", "Value"),
        (5, "Yaw", "Value"),
        (6, "Gripper", "State")
    ]
    
    for idx, (dim, title, ylabel) in enumerate(dimensions):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # plot ee_state
        ax.plot(time_steps[:-2], ee_state_euler[:-2, dim], label='EE State', linewidth=2)
        
        
        ax.plot(time_steps[:-2], trajectory_unwrapped[:-2, dim], label='Trajectory', linewidth=2, linestyle='--')
        

        if 3 <= dim <= 5:
            ax.set_title(f"{title} (Unwrapped)")
            # optional: add reference lines for the original angle range
            ax.axhline(y=np.pi, color='r', linestyle=':', alpha=0.3)
            ax.axhline(y=-np.pi, color='r', linestyle=':', alpha=0.3)
        
        ax.set_xlabel('Time Steps')
        ax.set_ylabel(ylabel)
        ax.grid(True)
        ax.legend()
        
        # add statistics information
        stats_text_ee_state = f'EE State: Mean: {np.mean(ee_state_euler[:, dim]):.3f}\nStd: {np.std(ee_state_euler[:, dim]):.3f}'
        stats_text_trajectory = f'Trajectory: Mean: {np.mean(trajectory_unwrapped[:, dim]):.3f}\nStd: {np.std(trajectory_unwrapped[:, dim]):.3f}'
        ax.text(0.02, 0.95, stats_text_ee_state, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.text(0.02, 0.75, stats_text_trajectory, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # save image
    plot_path = plot_dir / f"{Path(filepath).stem}_all_dimensions.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Plot saved to {plot_path}")

def process_dataset_directory(dataset_dir):
    """process all hdf5 files in the dataset directory"""
    dataset_dir = Path(dataset_dir)
    
    # get all hdf5 files
    hdf5_files = list(dataset_dir.glob("*.hdf5"))
    
    for filepath in hdf5_files:
        print(f"\nProcessing {filepath}")
        plot_trajectory_comparison(filepath)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str,default="/mnt/data/310_jiarui/VLABench/media/deterministic_v1/select_fruit_v2/select_fruit", help='Path to dataset directory')
    args = parser.parse_args()
    
    process_dataset_directory(args.dir)
