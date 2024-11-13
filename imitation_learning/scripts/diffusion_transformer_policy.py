import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import os

import sys
sys.path.append('/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/scripts')

from policy.transformer_for_diffusion import TransformerForDiffusion
from policy.diffusion_transformer import DiffusionTransformerLowdimPolicy
from policy.normalizer import LinearNormalizer
import policy.utils_transformer as utils
import wandb
import numpy as np
from datetime import datetime

# Dataset configuration
data_dir = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/data/diffusion'
num_episodes = 73
pred_horizon = 5
obs_horizon = 1
action_horizon = 3
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

# # Access data for fitting
# obs_data = dataset.normalized_train_data['obs']  # Check this variable name based on the dataset's actual structure
# action_data = dataset.normalized_train_data['action']

# # Initialize and fit the normalizer
# normalizer = LinearNormalizer()
# data_to_fit = {'obs': obs_data, 'action': action_data}  # Replace with actual tensors if necessary
# normalizer.fit(data_to_fit)

# # Confirm keys in normalizer
# print("Keys in normalizer.params_dict after fitting:", list(normalizer.params_dict.keys()))


normalizer = LinearNormalizer()
data_to_fit = {'obs': dataset[0]['obs'], 'action': dataset[0]['action']}
normalizer.fit(data_to_fit)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)


# # Initialize model components
# transformer_model = TransformerForDiffusion(input_dim=action_dim, output_dim=action_dim,
#                                             horizon=pred_horizon, n_obs_steps=obs_horizon, cond_dim=obs_dim * obs_horizon)

transformer_model = TransformerForDiffusion(
    input_dim=action_dim,
    output_dim=action_dim,
    horizon=pred_horizon,
    n_obs_steps=obs_horizon,
    cond_dim=obs_horizon*obs_dim,  # Ensure this is correctly set to 363 or as required
    n_layer=12,  # Keep this as per your settings
    n_head=12,
    n_emb=768,
    p_drop_emb=0.1,
    p_drop_attn=0.1,
    causal_attn=False,
    time_as_cond=True,
    obs_as_cond=True  # Assuming you're using observation as conditioning
)

noise_scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
                                clip_sample=True, prediction_type='epsilon')
# policy = DiffusionTransformerLowdimPolicy(
#     model=transformer_model, noise_scheduler=noise_scheduler, horizon=pred_horizon,
#     obs_dim=obs_dim, action_dim=action_dim, n_action_steps=action_horizon,
#     n_obs_steps=obs_horizon, obs_as_cond=True, pred_action_steps_only=False
# )

# Pass the fitted normalizer to the policy
policy = DiffusionTransformerLowdimPolicy(
    model=transformer_model,
    noise_scheduler=noise_scheduler,
    horizon=pred_horizon,
    obs_dim=obs_dim,
    action_dim=action_dim,
    n_action_steps=action_horizon,
    n_obs_steps=obs_horizon,
    obs_as_cond=True,
    pred_action_steps_only=False
)

# Set the fitted normalizer in the policy
policy.set_normalizer(normalizer)

policy.to(device='cuda' if torch.cuda.is_available() else 'cpu')


# Training parameters
num_epochs = 1000
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4, weight_decay=1e-6)
lr_scheduler = get_scheduler(
    name='cosine', optimizer=optimizer, num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs
)

user = "hisham-khalil"
project = "turtlebot_diffusion"
display_name = "experiment-2024-10-31"
config={
    "num_epochs": num_epochs,
    "batch_size": dataloader.batch_size,
    "noise_pred_net": "transformer"
    }

wandb.init(entity=user, project=project, name=display_name, config=config)

# Training Loop
device = policy.device
for epoch_idx in range(num_epochs):

    epoch_loss = list()

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
        epoch_loss.append(loss.item())

        wandb.log({"batch_loss": loss.item()})
    
    avg_epoch_loss = np.mean(epoch_loss)

    print(f'Epoch {epoch_idx+1} Loss: {avg_epoch_loss}')
    wandb.log({"epoch_loss": avg_epoch_loss, "epoch": epoch_idx})

wandb.finish()

# Save Model
save_dir = '/home/hisham246/uwaterloo/ME780/tb3_diffusion_models'
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
os.makedirs(save_dir, exist_ok=True)
torch.save(policy.state_dict(), os.path.join(save_dir, f'diffusion_transformer_policy_{timestamp}.pt'))