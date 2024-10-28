import numpy as np
import os
import json

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

# normalize data
def get_data_stats(data):
    data = data.reshape(-1,data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

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

def load_json_episodes(data_dir, num_episodes):
    train_data = {'obs': [], 'action': []}
    episode_lengths = []

    for i in range(num_episodes):
        file_path = os.path.join(data_dir, f'trajectory_{i}.json')
        
        with open(file_path, 'r') as f:
            episode_data = json.load(f)

        # Extract x, y, z, w from robot_pos
        robot_pos = np.array(episode_data['robot_pos'])
        x, y, z, w = robot_pos[:, 0], robot_pos[:, 1], robot_pos[:, 5], robot_pos[:, 6]

        # Calculate theta from z and w components
        theta = quaternion_to_yaw(z, w)

        # Create modified robot_pos with [x, y, theta]
        modified_robot_pos = np.column_stack((x, y, theta))

        # Convert laser_scan to numpy array and replace infinity values with 3.5
        laser_scan = np.array(episode_data['laser_scan'])
        laser_scan[np.isinf(laser_scan)] = 3.5

        # Concatenate modified_robot_pos and laser_scan
        obs = np.concatenate([modified_robot_pos, laser_scan], axis=1)
        
        # Convert robot_vel to numpy array
        action = np.array(episode_data['robot_vel'])

        # Add observations and actions to train_data
        train_data['obs'].append(obs)
        train_data['action'].append(action)

        # Store the length of the episode
        episode_lengths.append(len(obs))

    # Concatenate all episodes
    train_data['obs'] = np.concatenate(train_data['obs'], axis=0)
    train_data['action'] = np.concatenate(train_data['action'], axis=0)

    return train_data, episode_lengths

def sample_sequence(train_data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result