#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import os
import wandb
# from torch.utils.tensorboard import SummaryWriter

import sys
sys.path.append('/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/scripts')

import utils
import unet

# # Directory to save TensorBoard logs
# log_dir = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/logs'
# os.makedirs(log_dir, exist_ok=True)

# # Initialize TensorBoard writer
# writer = SummaryWriter(log_dir=log_dir)



class TurtleBot3Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, num_episodes, pred_horizon, obs_horizon, action_horizon):
        # Load data from JSON files
        train_data, episode_lengths = utils.load_json_episodes(data_dir, num_episodes)

        # Compute start and end of each state-action sequence, handle padding
        indices = utils.create_sample_indices(
            episode_lengths=episode_lengths,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1
        )

        # Compute statistics and normalize data to [-1, 1]
        stats = dict()
        normalized_train_data = dict()
        for key, data in train_data.items():
            stats[key] = utils.get_data_stats(data)
            normalized_train_data[key] = utils.normalize_data(data, stats[key])

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = self.indices[idx]

        # Get normalized data using these indices
        nsample = utils.sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # Discard unused observations
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :]
        return nsample

# Example usage:
data_dir = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/data/diffusion'  # Replace with your JSON directory
num_episodes = 73
pred_horizon = 10
obs_horizon = 2
action_horizon = 5

dataset = TurtleBot3Dataset(data_dir, num_episodes, pred_horizon, obs_horizon, action_horizon)
# print(dataset.normalized_train_data['obs'].shape)

dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=256, 
    num_workers=1,
    shuffle=True,
    pin_memory=True,
    persistent_workers=True)

batch = next(iter(dataloader))
print("batch['obs'].shape:", batch['obs'].shape)
print("batch['action'].shape", batch['action'].shape)

# observation and action dimensions
obs_dim = 363
action_dim = 2

# create network object
noise_pred_net = unet.ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon
)

# example inputs
noised_action = torch.randn((1, pred_horizon, action_dim))
obs = torch.zeros((1, obs_horizon, obs_dim))
diffusion_iter = torch.zeros((1,))

# the noise prediction network
# takes noisy action, diffusion iteration and observation as input
# predicts the noise added to action
noise = noise_pred_net(
    sample=noised_action,
    timestep=diffusion_iter,
    global_cond=obs.flatten(start_dim=1))

# illustration of removing noise
# the actual noise removal is performed by NoiseScheduler
# and is dependent on the diffusion noise schedule

# Ensure matching shapes between noised_action and noise
if noised_action.shape[1] != noise.shape[1]:
    min_len = min(noised_action.shape[1], noise.shape[1])
    noised_action = noised_action[:, :min_len, :]
    noise = noise[:, :min_len, :]

# Perform the subtraction

denoised_action = noised_action - noise

# for this demo, we use DDPMScheduler with 100 diffusion iterations
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
_ = noise_pred_net.to(device)

num_epochs = 200

# Exponential Moving Average
# accelerates training and improves stability
# holds a copy of the model weights
ema = EMAModel(
    parameters=noise_pred_net.parameters(),
    power=0.75)

# Standard ADAM optimizer
# Note that EMA parametesr are not optimized
optimizer = torch.optim.AdamW(
    params=noise_pred_net.parameters(),
    lr=1e-4, weight_decay=1e-6)

# Cosine LR schedule with linear warmup
lr_scheduler = get_scheduler(
    name='cosine',
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

# Initialize a WandB project

user = "hisham-khalil"
project = "turtlebot_diffusion"
display_name = "experiment-2024-10-31"
config={
    "num_epochs": num_epochs,
    "batch_size": dataloader.batch_size,
    "learning_rate": 1e-4,
    "weight_decay": 1e-6,
    "scheduler": "cosine"
    }

wandb.init(entity=user, project=project, name=display_name, config=config)


with tqdm(range(num_epochs), desc='Epoch') as tglobal:
    # epoch loop
    for epoch_idx in tglobal:
        epoch_loss = list()
        # batch loop
        with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
            for nbatch in tepoch:
                # data normalized in dataset
                # device transfer
                nobs = nbatch['obs'].to(device)
                naction = nbatch['action'].to(device)
                B = nobs.shape[0]

                # observation as FiLM conditioning
                # (B, obs_horizon, obs_dim)
                obs_cond = nobs[:, :obs_horizon, :].contiguous().view(nobs.size(0), -1).float()

                # (B, obs_horizon * obs_dim)
                obs_cond = obs_cond.flatten(start_dim=1)

                # print(f"Shape of obs_cond (batch): {obs_cond.shape}")

                # sample noise to add to actions
                noise = torch.randn(naction.shape, device=device)

                # sample a diffusion iteration for each data point
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (B,), device=device
                ).long()

                # add noise to the clean images according to the noise magnitude at each diffusion iteration
                # (this is the forward diffusion process)
                noisy_actions = noise_scheduler.add_noise(
                    naction, noise, timesteps)
                
                # Convert tensors to float before passing to the model
                noisy_actions = noisy_actions.float()
                obs_cond = obs_cond.float()
                timesteps = timesteps.float()

                # predict the noise residual
                noise_pred = noise_pred_net(
                    noisy_actions, timesteps, global_cond=obs_cond)

                # L2 loss
                loss = nn.functional.mse_loss(noise_pred, noise)

                # optimize
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # step lr scheduler every batch
                # this is different from standard pytorch behavior
                lr_scheduler.step()

                # update Exponential Moving Average of the model weights
                ema.step(noise_pred_net.parameters())

                wandb.log({"batch_loss": loss.item()})

                # logging
                loss_cpu = loss.item()
                epoch_loss.append(loss_cpu)
                tepoch.set_postfix(loss=loss_cpu)


        # Log epoch loss to WandB
        avg_epoch_loss = np.mean(epoch_loss)
        wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch_idx})
        tglobal.set_postfix(loss=avg_epoch_loss)

# Weights of the EMA model
# is used for inference
ema_noise_pred_net = noise_pred_net
ema.copy_to(ema_noise_pred_net.parameters())

wandb.finish()

# Directory to save the model
save_dir = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/models/diffusion'

# Save the trained model's state_dict
torch.save(noise_pred_net.state_dict(), os.path.join(save_dir, 'noise_pred_net.pt'))

# Save the EMA model's state_dict for inference
torch.save(ema_noise_pred_net.state_dict(), os.path.join(save_dir, 'ema_noise_pred_net.pt'))