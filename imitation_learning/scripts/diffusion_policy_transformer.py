#!/usr/bin/env python

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import os

from policy.transformer_for_diffusion import TransformerForDiffusion
from policy.diffusion_transformer import DiffusionTransformerLowdimPolicy
from policy.normalizer import LinearNormalizer
import policy.utils_transformer as utils

# Dataset configuration
data_dir = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/data/diffusion'
num_episodes = 73
pred_horizon = 10
obs_horizon = 5
action_horizon = 5
obs_dim = 363
action_dim = 2

class TurtleBot3Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_episodes, pred_horizon, obs_horizon, action_horizon):
        # Load data from JSON files
        train_data, episode_lengths = utils.load_json_episodes(data_dir, num_episodes)
        
        # Verify keys in train_data
        required_keys = ['obs', 'action']
        missing_keys = [key for key in required_keys if key not in train_data]
        if missing_keys:
            raise KeyError(f"Missing required keys in train_data: {missing_keys}")

        # Initialize and fit the normalizer on obs and action data
        self.normalizer = LinearNormalizer()
        print("Fitting normalizer on keys:", list(train_data.keys()))
        self.normalizer.fit({'obs': train_data['obs'], 'action': train_data['action']}, mode='limits')

        # Normalize the loaded data
        self.normalized_train_data = self.normalizer.normalize(train_data)

        # Generate sampling indices
        self.indices = utils.create_sample_indices(
            episode_lengths=episode_lengths,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1
        )

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        # Extract normalized sequences
        nsample = utils.sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # Trim observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]
        return nsample


# Initialize DataLoader
dataset = TurtleBot3Dataset(data_dir, num_episodes, pred_horizon, obs_horizon, action_horizon)

# Access data for fitting
obs_data = dataset.normalized_train_data['obs']  # Check this variable name based on the dataset's actual structure
action_data = dataset.normalized_train_data['action']

# Initialize and fit the normalizer
normalizer = LinearNormalizer()
data_to_fit = {'obs': obs_data, 'action': action_data}  # Replace with actual tensors if necessary
normalizer.fit(data_to_fit)

# Confirm keys in normalizer
print("Keys in normalizer.params_dict after fitting:", list(normalizer.params_dict.keys()))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# Initialize model components
transformer_model = TransformerForDiffusion(input_dim=action_dim, output_dim=action_dim,
                                            horizon=pred_horizon, n_obs_steps=obs_horizon, cond_dim=obs_dim * obs_horizon)
noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                                clip_sample=True, prediction_type='epsilon')
policy = DiffusionTransformerLowdimPolicy(
    model=transformer_model, noise_scheduler=noise_scheduler, horizon=pred_horizon,
    obs_dim=obs_dim, action_dim=action_dim, n_action_steps=action_horizon,
    n_obs_steps=obs_horizon, obs_as_cond=True, pred_action_steps_only=False
)
policy.to(device='cuda' if torch.cuda.is_available() else 'cpu')

# Training parameters
num_epochs = 100
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name='cosine', optimizer=optimizer, num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

# Training Loop
device = policy.device
for epoch_idx in range(num_epochs):
    epoch_loss = 0
    for batch in tqdm(dataloader, desc=f'Epoch {epoch_idx+1}/{num_epochs}'):
        obs = batch['obs'].to(device)
        action = batch['action'].to(device)

        # Prepare input batch
        batch_data = {'obs': obs, 'action': action}
        loss = policy.compute_loss(batch_data)  # Directly use policy's compute_loss

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Accumulate loss for logging
        epoch_loss += loss.item()
    print(f'Epoch {epoch_idx+1} Loss: {epoch_loss / len(dataloader)}')

# Save Model
save_dir = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/models/diffusion'
os.makedirs(save_dir, exist_ok=True)
torch.save(policy.state_dict(), os.path.join(save_dir, 'diffusion_transformer_policy.pt'))


# import torch
# import torch.nn as nn
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.optimization import get_scheduler
# from tqdm.auto import tqdm
# import os
# import numpy as np

# import sys
# sys.path.append('/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/scripts')

# from policy.transformer_for_diffusion import TransformerForDiffusion
# from policy.diffusion_transformer import DiffusionTransformerLowdimPolicy
# import utils
# from policy.normalizer import LinearNormalizer 


# # Dataset and training parameters
# data_dir = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/data/diffusion'
# num_episodes = 73
# pred_horizon = 10
# obs_horizon = 5
# action_horizon = 5
# obs_dim = 363
# action_dim = 2

# class TurtleBot3Dataset(torch.utils.data.Dataset):
#     def __init__(self, data_dir, num_episodes, pred_horizon, obs_horizon, action_horizon):
#         # Load data from JSON files
#         train_data, episode_lengths = utils.load_json_episodes(data_dir, num_episodes)

#         # Compute start and end of each state-action sequence, handle padding
#         indices = utils.create_sample_indices(
#             episode_lengths=episode_lengths,
#             sequence_length=pred_horizon,
#             pad_before=obs_horizon - 1,
#             pad_after=action_horizon - 1
#         )

#         self.indices = indices
#         self.train_data = train_data
#         self.pred_horizon = pred_horizon
#         self.action_horizon = action_horizon
#         self.obs_horizon = obs_horizon

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx):
#         buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

#         # Fetch data sample
#         sample = utils.sample_sequence(
#             train_data=self.train_data,
#             sequence_length=self.pred_horizon,
#             buffer_start_idx=buffer_start_idx,
#             buffer_end_idx=buffer_end_idx,
#             sample_start_idx=sample_start_idx,
#             sample_end_idx=sample_end_idx
#         )

#         # Extract and limit observations for obs_horizon
#         sample['obs'] = sample['obs'][:self.obs_horizon, :]
#         return sample

# dataset = TurtleBot3Dataset(data_dir, num_episodes, pred_horizon, obs_horizon, action_horizon)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# # visualize data in batch
# batch = next(iter(dataloader))
# print("batch['obs'].shape:", batch['obs'].shape)
# print("batch['action'].shape", batch['action'].shape)

# # Initialize LinearNormalizer and fit it to the dataset stats
# normalizer = LinearNormalizer()
# stats = {
#     'obs': np.vstack([sample['obs'] for sample in dataset]),
#     'action': np.vstack([sample['action'] for sample in dataset])
# }
# normalizer.fit(stats)

# # Normalize the entire dataset before training
# for sample in dataset:
#     sample['obs'] = normalizer['obs'].normalize(sample['obs'])
#     sample['action'] = normalizer['action'].normalize(sample['action'])

# # Initialize the transformer model and diffusion scheduler
# transformer_model = TransformerForDiffusion(
#     input_dim=action_dim,
#     output_dim=action_dim,
#     horizon=pred_horizon,
#     n_obs_steps=obs_horizon,
#     cond_dim=obs_dim * obs_horizon  
# )
# noise_scheduler = DDPMScheduler(
#     num_train_timesteps=100,
#     beta_schedule='squaredcos_cap_v2',
#     clip_sample=True,
#     prediction_type='epsilon'
# )

# # Initialize the diffusion policy
# noise_pred_net = DiffusionTransformerLowdimPolicy(
#     model=transformer_model,
#     noise_scheduler=noise_scheduler,
#     horizon=pred_horizon,
#     obs_dim=obs_dim,
#     action_dim=action_dim,
#     n_action_steps=action_horizon,
#     n_obs_steps=obs_horizon,
#     obs_as_cond=True,
#     pred_action_steps_only=False
# )

# # Move policy to device
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# noise_pred_net.to(device)

# # Set up optimizer, learning rate scheduler, and EMA
# num_epochs = 100
# optimizer = torch.optim.AdamW(noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6)
# lr_scheduler = get_scheduler(
#     name='cosine',
#     optimizer=optimizer,
#     num_warmup_steps=500,
#     num_training_steps=len(dataloader) * num_epochs
# )
# ema = torch.optim.swa_utils.AveragedModel(noise_pred_net)

# with tqdm(range(num_epochs), desc='Epoch') as tglobal:
#     # epoch loop
#     for epoch_idx in tglobal:
#         epoch_loss = list()
#         # batch loop
#         with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
#             for nbatch in tepoch:
#                 # Device transfer
#                 nobs = normalizer['obs'].normalize(nbatch['obs']).to(device)
#                 naction = normalizer['action'].normalize(nbatch['action']).to(device)
#                 B = nobs.shape[0]

#                 # Observation as FiLM conditioning
#                 obs_cond = nobs[:, :obs_horizon, :].contiguous().view(nobs.size(0), -1).float()

#                 # Flatten observation conditioning
#                 obs_cond = obs_cond.flatten(start_dim=1)

#                 # Sample noise to add to actions
#                 noise = torch.randn(naction.shape, device=device)

#                 # Sample diffusion timestep for each data point
#                 timesteps = torch.randint(
#                     0, noise_scheduler.config.num_train_timesteps,
#                     (B,), device=device
#                 ).long()

#                 # Add noise to actions (forward diffusion)
#                 noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)
                
#                 # Convert tensors to float before passing to the model
#                 noisy_actions = noisy_actions.float()
#                 obs_cond = obs_cond.float()
#                 timesteps = timesteps.float()

#                 # Predict the noise residual
#                 noise_pred = noise_pred_net(
#                     noisy_actions, timesteps, cond=obs_cond)

#                 # L2 loss
#                 loss = nn.functional.mse_loss(noise_pred, noise)

#                 # Optimize
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 # Step the LR scheduler every batch
#                 lr_scheduler.step()

#                 # Update Exponential Moving Average of the model weights
#                 ema.update_parameters(noise_pred_net)

#                 # Logging
#                 loss_cpu = loss.item()
#                 epoch_loss.append(loss_cpu)
#                 tepoch.set_postfix(loss=loss_cpu)
#         tglobal.set_postfix(loss=np.mean(epoch_loss))

# # Save the model
# save_dir = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/models/diffusion'
# os.makedirs(save_dir, exist_ok=True)
# torch.save(noise_pred_net.state_dict(), os.path.join(save_dir, 'diffusion_transformer_policy.pt'))
# torch.save(ema.module.state_dict(), os.path.join(save_dir, 'ema_diffusion_transformer_policy.pt'))
