import numpy as np
import os
import json
import torch

def quaternion_to_yaw(z, w):
    """
    Convert quaternion to yaw angle (theta).
    Args:
        z (float): The z-component of the quaternion.
        w (float): The w-component of the quaternion.
    Returns:
        theta (float): The yaw angle.
    """
    theta = 2 * np.arctan2(z, w)
    return theta

def create_sample_indices(episode_lengths, sequence_length, pad_before=0, pad_after=0):
    indices = []
    start_idx = 0
    
    # Iterate over each episode's length
    for episode_length in episode_lengths:
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx
            ])
        
        start_idx += episode_length  # Update start_idx for the next episode

    indices = np.array(indices)
    return indices

def sample_sequence(train_data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        
        # Initialize an empty array with zeros if padding is needed
        data = torch.zeros(
            (sequence_length,) + sample.shape[1:],
            dtype=sample.dtype
        )

        # Place the actual sample in the zero-initialized array
        data[sample_start_idx:sample_end_idx] = sample.clone().detach()

        # Apply edge padding for sequences that require it
        if sample_start_idx > 0:
            data[:sample_start_idx] = data[sample_start_idx]
        if sample_end_idx < sequence_length:
            data[sample_end_idx:] = data[sample_end_idx - 1]
        
        result[key] = data
    return result


# Ensure to use correct dtype in `np.array()` calls in `load_json_episodes()`
def load_json_episodes(data_dir, num_episodes):
    train_data = {'obs': [], 'action': []}
    episode_lengths = []

    for i in range(num_episodes):
        file_path = os.path.join(data_dir, f'trajectory_{i}.json')
        
        with open(file_path, 'r') as f:
            episode_data = json.load(f)

        # Extract and process observation and action data as before
        robot_pos = np.array(episode_data['robot_pos'], dtype=np.float32)
        x, y, z, w = robot_pos[:, 0], robot_pos[:, 1], robot_pos[:, 5], robot_pos[:, 6]
        theta = quaternion_to_yaw(z, w)
        modified_robot_pos = np.column_stack((x, y, theta))
        
        laser_scan = np.array(episode_data['laser_scan'], dtype=np.float32)
        laser_scan[np.isinf(laser_scan)] = 3.5  # Clip infinite values
        obs = np.concatenate([modified_robot_pos, laser_scan], axis=1)
        
        action = np.array(episode_data['robot_vel'], dtype=np.float32)

        train_data['obs'].append(obs)
        train_data['action'].append(action)
        episode_lengths.append(len(obs))

    train_data['obs'] = np.concatenate(train_data['obs'], axis=0)
    train_data['action'] = np.concatenate(train_data['action'], axis=0)

    return train_data, episode_lengths