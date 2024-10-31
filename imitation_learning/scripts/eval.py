import numpy as np
import torch
import utils
import unet
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.optimization import get_scheduler

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

# Ensure you have the trained model loaded
ckpt_path = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/models/diffusion/ema_noise_pred_net.pth'
ema_noise_pred_net = unet.ConditionalUnet1D(
    input_dim=2,  # action_dim
    global_cond_dim=5 * 363  # obs_horizon * obs_dim (for example)
)
ema_noise_pred_net.load_state_dict(torch.load(ckpt_path, map_location='cuda'))
ema_noise_pred_net.to('cuda')
ema_noise_pred_net.eval()

# Load the dataset
data_dir = '/home/hisham246/uwaterloo/ME780/turtlebot_ws/src/ImitationLearning-Turtlebot3/imitation_learning/data/diffusion'
num_episodes = 1  # Load a single episode for testing
pred_horizon = 10
obs_horizon = 5
action_horizon = 5

# Initialize the dataset and dataloader
dataset = TurtleBot3Dataset(data_dir, num_episodes, pred_horizon, obs_horizon, action_horizon)
dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=1,  # Single sample for testing
    num_workers=0,
    shuffle=False
)

# Get one batch from the dataloader
batch = next(iter(dataloader))

# Extract observations and actions from the batch
obs = batch['obs'].to('cuda', dtype=torch.float32)  # [1, obs_horizon, obs_dim]
true_actions = batch['action'].to('cuda', dtype=torch.float32)  # [1, pred_horizon, action_dim]
print(true_actions)

# Flatten observation to match the model's expected input
obs_cond = obs.flatten(start_dim=1)  # [1, obs_horizon * obs_dim]

# Initialize action from Gaussian noise
B = 1  # Batch size
pred_horizon = true_actions.shape[1]
action_dim = true_actions.shape[2]
noisy_action = torch.randn((B, pred_horizon, action_dim), device='cuda')
naction = noisy_action

# Set timesteps for the scheduler
num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    beta_schedule='squaredcos_cap_v2',
    clip_sample=True,
    prediction_type='epsilon'
)
noise_scheduler.set_timesteps(num_diffusion_iters)

# Diffusion denoising loop
with torch.no_grad():
    for k in noise_scheduler.timesteps:
        noise_pred = ema_noise_pred_net(
            sample=naction,
            timestep=k,
            global_cond=obs_cond
        )

        # Inverse diffusion step (remove noise)
        naction = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=naction
        ).prev_sample

# Unnormalize the predicted actions
pred_actions = utils.unnormalize_data(naction.cpu().numpy(), stats=dataset.stats['action'])

# Compare predicted actions with ground truth
pred_actions = pred_actions.squeeze(0)  # Remove batch dimension
true_actions = true_actions.squeeze(0).cpu().numpy()

# Calculate Mean Squared Error (MSE) between predicted and true actions
mse = np.mean((pred_actions - true_actions) ** 2)
print(f"Predicted Actions: {pred_actions}")
print(f"True Actions: {true_actions}")
print(f"Mean Squared Error: {mse}")