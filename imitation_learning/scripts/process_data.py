# import numpy as np
# import json

# file_path = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/data/supervised/trajectory_3.json'

# # Open the JSON file and load it as a dictionary
# with open(file_path, 'r') as file:
#     data_dict = json.load(file)

# print(len(data_dict['laser_scan']))

# import torch
# import numpy as np
# import os
# import json

# # normalize data
# def get_data_stats(data):
#     data = data.reshape(-1,data.shape[-1])
#     stats = {
#         'min': np.min(data, axis=0),
#         'max': np.max(data, axis=0)
#     }
#     return stats

# def normalize_data(data, stats):
#     # nomalize to [0,1]
#     ndata = (data - stats['min']) / (stats['max'] - stats['min'])
#     # normalize to [-1, 1]
#     ndata = ndata * 2 - 1
#     return ndata

# def unnormalize_data(ndata, stats):
#     ndata = (ndata + 1) / 2
#     data = ndata * (stats['max'] - stats['min']) + stats['min']
#     return data
# def create_sample_indices(episode_lengths, sequence_length, pad_before=0, pad_after=0):
#     indices = []
#     start_idx = 0
    
#     # Iterate over each episode's length
#     for episode_length in episode_lengths:
#         min_start = -pad_before
#         max_start = episode_length - sequence_length + pad_after

#         for idx in range(min_start, max_start + 1):
#             buffer_start_idx = max(idx, 0) + start_idx
#             buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
#             start_offset = buffer_start_idx - (idx + start_idx)
#             end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
#             sample_start_idx = 0 + start_offset
#             sample_end_idx = sequence_length - end_offset
#             indices.append([
#                 buffer_start_idx, buffer_end_idx,
#                 sample_start_idx, sample_end_idx
#             ])
        
#         start_idx += episode_length  # Update start_idx for the next episode

#     indices = np.array(indices)
#     return indices

# def load_json_episodes(data_dir, num_episodes):
#     train_data = {'obs': [], 'action': []}
#     episode_lengths = []

#     for i in range(num_episodes):
#         file_path = os.path.join(data_dir, f'trajectory_{i}.json')
        
#         with open(file_path, 'r') as f:
#             episode_data = json.load(f)

#         # Concatenate observations and actions from the current episode
#         robot_pos = np.array(episode_data['robot_pos'])[:, [0, 1, 5, 6]]
#         laser_scan = np.array(episode_data['laser_scan'])

#         # Replace infinity values in laser scan with 3.5
#         laser_scan[np.isinf(laser_scan)] = 3.5


#         obs = np.concatenate([robot_pos, laser_scan], axis=1)
#         print(obs)
#         action = np.array(episode_data['robot_vel'])

#         train_data['obs'].append(obs)
#         train_data['action'].append(action)
#         episode_lengths.append(len(obs))  # Store the length of the episode

#     # Concatenate all episodes
#     train_data['obs'] = np.concatenate(train_data['obs'], axis=0)
#     train_data['action'] = np.concatenate(train_data['action'], axis=0)

#     return train_data, episode_lengths

# def sample_sequence(train_data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx):
#     result = dict()
#     for key, input_arr in train_data.items():
#         sample = input_arr[buffer_start_idx:buffer_end_idx]
#         data = sample
#         if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
#             data = np.zeros(
#                 shape=(sequence_length,) + input_arr.shape[1:],
#                 dtype=input_arr.dtype)
#             if sample_start_idx > 0:
#                 data[:sample_start_idx] = sample[0]
#             if sample_end_idx < sequence_length:
#                 data[sample_end_idx:] = sample[-1]
#             data[sample_start_idx:sample_end_idx] = sample
#         result[key] = data
#     return result

# class TurtleBot3Dataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir, num_episodes, pred_horizon, obs_horizon, action_horizon):
#         # Load data from JSON files
#         train_data, episode_lengths = load_json_episodes(data_dir, num_episodes)

#         # Compute start and end of each state-action sequence, handle padding
#         indices = create_sample_indices(
#             episode_lengths=episode_lengths,
#             sequence_length=pred_horizon,
#             pad_before=obs_horizon - 1,
#             pad_after=action_horizon - 1
#         )

#         # Compute statistics and normalize data to [-1, 1]
#         stats = dict()
#         normalized_train_data = dict()
#         for key, data in train_data.items():
#             stats[key] = get_data_stats(data)
#             normalized_train_data[key] = normalize_data(data, stats[key])

#         self.indices = indices
#         self.stats = stats
#         self.normalized_train_data = normalized_train_data
#         self.pred_horizon = pred_horizon
#         self.action_horizon = action_horizon
#         self.obs_horizon = obs_horizon

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

#         # Get normalized data using these indices
#         nsample = sample_sequence(
#             train_data=self.normalized_train_data,
#             sequence_length=self.pred_horizon,
#             buffer_start_idx=buffer_start_idx,
#             buffer_end_idx=buffer_end_idx,
#             sample_start_idx=sample_start_idx,
#             sample_end_idx=sample_end_idx
#         )

#         # Discard unused observations
#         nsample['obs'] = nsample['obs'][:self.obs_horizon, :]
#         return nsample

# # Example usage:
# data_dir = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/data/diffusion'  # Replace with your JSON directory
# num_episodes = 25
# pred_horizon = 10
# obs_horizon = 5
# action_horizon = 5

# dataset = TurtleBot3Dataset(data_dir, num_episodes, pred_horizon, obs_horizon, action_horizon)

# dataloader = torch.utils.data.DataLoader(
#     dataset, 
#     batch_size=256, 
#     shuffle=True,
#     pin_memory=True,
#     persistent_workers=True)

# batch = next(iter(dataloader))
# # print("batch['obs'].shape:", batch['obs'].shape)
# # print("batch['action'].shape", batch['action'].shape)

