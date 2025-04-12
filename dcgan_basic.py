import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.cuda.amp as amp  # For mixed precision training
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip  # Modified import

# Simple dummy reward functions to replace complex ones
def get_lambda(t, total_epochs):
    """Calculate dynamic weight based on training progress"""
    if t < total_epochs * 0.3:  # Early stage
        return 0.8
    elif t < total_epochs * 0.7:  # Middle stage
        return 0.5
    else:  # Late stage
        return 0.3

def calculate_rewards(epoch, total_epochs, current_fid, previous_fid, current_loss_g, current_loss_d, diversity_score):
    """
    Calculate mixed rewards with dynamic weights
    
    Args:
        epoch: current epoch
        total_epochs: total number of epochs
        current_fid: current FID score
        previous_fid: FID score from previous epoch
        current_loss_g: current generator loss
        current_loss_d: current discriminator loss
        diversity_score: score indicating image diversity
    """
    # Calculate improvement reward
    fid_improvement = previous_fid - current_fid  # Lower FID is better
    loss_balance = max(0.1, abs(current_loss_g - current_loss_d))  # Prevent division by zero
    r_improve = 0.5 * fid_improvement + 0.3 * (1/loss_balance) + 0.2 * diversity_score
    
    # Calculate level reward
    r_level = 0.4 * (1/(current_fid + 0.1)) + 0.3 * (1/(current_loss_g + 0.1)) + 0.3 * diversity_score
    
    # Get dynamic weight
    lambda_t = get_lambda(epoch, total_epochs)
    
    # Calculate total reward
    total_reward = lambda_t * r_improve + (1 - lambda_t) * r_level
    
    return total_reward

def calculate_diversity_score(generated_images):
    # Simple diversity calculation
    if isinstance(generated_images, torch.Tensor):
        batch_size = generated_images.size(0)
        if batch_size <= 1:
            return 0.0
        # Just calculate variance across batch
        return torch.var(generated_images.view(batch_size, -1), dim=0).mean().item()
    return 0.0

# Simple dummy FID calculator
class FIDCalculator:
    def __init__(self, device):
        self.device = device
        
    def calculate_fid(self, real_images, fake_images):
        # Simple placeholder for FID
        # Just calculate mean squared difference between real and fake
        if isinstance(real_images, torch.Tensor) and isinstance(fake_images, torch.Tensor):
            real_mean = real_images.mean().item()
            fake_mean = fake_images.mean().item()
            return abs(real_mean - fake_mean) * 10.0
        return 100.0  # Default value

# Simple RL agent
class PPOHyperparameterOptimizer:
    def __init__(self, state_dim=4, action_dim=2):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.gamma = 0.99
        self.epsilon = 0.2
        self.learning_rate = 0.0003
        self.batch_size = 64
        
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Output range [-1, 1]
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
    def get_action(self, state):
        # state가 이미 (1, 4) shape일 것으로 가정하므로, unsqueeze(0) 없이 변환합니다.
        state = torch.FloatTensor(state)  # shape: (1, 4)
        with torch.no_grad():
            action_mean = self.actor(state)  # shape: (1, 2)
            action_std = torch.ones_like(action_mean) * 0.1
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()           # shape: (1, 2)
            log_prob = dist.log_prob(action).sum(dim=-1)
        
        # Scale actions to appropriate ranges; 
        # 첫 번째는 학습률, 두 번째는 beta1 값을 위해 올바른 인덱스를 사용합니다.
        scaled_action = torch.tensor([
            action[0, 0] * 0.0001 + 0.0002,  # learning rate: ~0.0001 to 0.0003
            action[0, 1] * 0.2 + 0.5         # beta1: 0.3 to 0.7
        ])
        
        # Convert to numpy array and ensure it's 1D
        scaled_action_np = scaled_action.detach().cpu().numpy().flatten()
        
        return scaled_action_np, log_prob.item()
    
    def store_transition(self, state, action, reward, log_prob):
        # Convert state and action to tensors, ensuring correct dimensions
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)  # Add batch dimension
        reward_tensor = torch.FloatTensor([reward])
        log_prob_tensor = torch.FloatTensor([log_prob])
        
        self.memory.append((state_tensor, action_tensor, reward_tensor, log_prob_tensor))
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
            
        # Sample a batch from memory
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        # Convert to numpy array first, then to tensor
        states = torch.FloatTensor(np.array([b[0] for b in batch]))
        actions = torch.FloatTensor(np.array([b[1] for b in batch]))
        rewards = torch.FloatTensor(np.array([b[2] for b in batch])).squeeze()  # shape: (batch_size,)
        old_log_probs = torch.FloatTensor(np.array([b[3] for b in batch]))
        
        # Calculate advantages
        with torch.no_grad():
            values = self.critic(states)  # shape: (batch_size, 1)
            advantages = rewards - values.squeeze()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Update actor
        for _ in range(3):  # Multiple epochs for actor update
            action_means = self.actor(states)
            action_stds = torch.ones_like(action_means) * 0.1
            dist = torch.distributions.Normal(action_means, action_stds)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        
        # Update critic
        for _ in range(3):  # Multiple epochs for critic update
            values = self.critic(states)  # shape: (batch_size, 1)
            critic_loss = nn.MSELoss()(values.squeeze(), rewards)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        # Clear memory after training
        self.memory = []

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
DATA_DIR = os.path.join(BASE_DIR, 'img_align_celeba')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set device (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def show_generated_images(images, num_images=64):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Generated Images")
    images = vutils.make_grid(images[:num_images], padding=2, normalize=True)
    images = np.transpose(images, (1, 2, 0))
    plt.imshow(images)
    plt.show()

def save_generated_images(images, num_images, epoch, idx):
    plt.figure(figsize=(10, 10))
    plt.axis("off")
    plt.title("Generated Images")
    images = vutils.make_grid(images[:num_images], padding=2, normalize=True)
    images = np.transpose(images, (1, 2, 0))
    fname = os.path.join(OUTPUT_DIR, f'fake_samples_epoch_{epoch}_batch_{idx}.png')
    plt.imsave(fname, images.numpy())
    plt.close()

def create_training_video(output_dir, fps=10):
    """
    Create a video from the generated images during training
    
    Args:
        output_dir: Directory containing the generated images
        fps: Frames per second for the output video
    """
    # Get all generated image files
    image_files = sorted([f for f in os.listdir(output_dir) 
                         if f.startswith('fake_samples_epoch_') and f.endswith('.png')])
    
    if not image_files:
        print("No generated images found for video creation")
        return
    
    # Create video clip
    clip = ImageSequenceClip([os.path.join(output_dir, img) for img in image_files], fps=fps)
    
    # Save video
    video_path = os.path.join(output_dir, 'training_progress.mp4')
    clip.write_videofile(video_path, codec='libx264')
    print(f"Training progress video saved to: {video_path}")

def plot_training_progress(log_file):
    """
    Plot training progress from log file
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    
    # Read log file
    df = pd.read_csv(log_file)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot losses
    ax1.plot(df['Batch'], df['Loss_D'], label='Discriminator Loss')
    ax1.plot(df['Batch'], df['Loss_G'], label='Generator Loss')
    ax1.set_title('Generator and Discriminator Losses')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot FID score
    ax2.plot(df['Batch'], df['FID'])
    ax2.set_title('FID Score')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('FID')
    ax2.grid(True)
    
    # Plot reward
    ax3.plot(df['Batch'], df['Reward'])
    ax3.set_title('Reward')
    ax3.set_xlabel('Batch')
    ax3.set_ylabel('Reward')
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_progress.png'))
    plt.close()

# Create a custom dataset class for CelebA
class CelebADataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Try different possible directory structures
        possible_dirs = [
            os.path.join(root_dir, 'img_align_celeba', 'img_align_celeba'),
            os.path.join(root_dir, 'img_align_celeba'),
            os.path.join(root_dir, 'celeba', 'img_align_celeba')
        ]
        
        self.image_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                self.image_dir = dir_path
                break
        
        if self.image_dir is None:
            raise FileNotFoundError(f"Could not find image directory. Tried: {possible_dirs}")
        
        print(f"Using image directory: {self.image_dir}")
        
        # Get all image files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        if not self.image_files:
            raise ValueError(f"No .jpg images found in {self.image_dir}")
        
        print(f"Found {len(self.image_files)} images in {self.image_dir}")
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, 0  # Return 0 as dummy label

def train_dcgan():
    # Set batch size to 32
    batch_size = 32
    image_size = 64  # Reduce image size
    
    # Initial hyperparameters
    current_params = {
        'lr': 0.0002,
        'beta1': 0.5
    }

    # Data Preparation - use smaller images
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        raise FileNotFoundError(f"Data directory not found: {DATA_DIR}. Please make sure the CelebA dataset is properly downloaded and placed in the correct location.")
    
    print(f"Data directory exists: {DATA_DIR}")
    
    dataset = CelebADataset(root_dir=DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Create the generator and the discriminator - much simpler models
    class Generator(nn.Module):
        def __init__(self, nz=100, ngf=64, nc=3):
            super(Generator, self).__init__()
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        def forward(self, input):
            return self.main(input)

    class Discriminator(nn.Module):
        def __init__(self, nc=3, ndf=64):
            super(Discriminator, self).__init__()
            # Main feature extraction layers
            self.features = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
            )
            
            # Final classification layer
            self.classifier = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
            )

        def forward(self, input):
            # Run through feature extraction
            features = self.features(input)
            # Run through classifier and flatten
            output = self.classifier(features)
            # Flatten output to match batch size
            return output.view(input.size(0))

    # Initialize models
    netG = Generator().to(device)
    netD = Discriminator().to(device)

    # Apply the weights_init function to randomly initialize all weights
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Initialize RL agent
    rl_agent = PPOHyperparameterOptimizer()
    
    # Training
    print(f"Using device: {device}")
    
    # Set up mixed precision training
    use_amp = True if device.type == 'cuda' else False
    if use_amp:
        scaler = amp.GradScaler()
    
    # Initialize FID calculator
    fid_calculator = FIDCalculator(device)

    # Use BCEWithLogitsLoss instead of BCELoss for numerical stability
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizerD = optim.Adam(netD.parameters(), lr=current_params['lr'], betas=(current_params['beta1'], 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=current_params['lr'], betas=(current_params['beta1'], 0.999))

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    num_epochs = 50
    fixed_noise = torch.randn(8, 100, 1, 1, device=device)

    # Initialize variables for reward calculation
    previous_fid = float('inf')
    best_fid = float('inf')
    previous_loss_g = float('inf')
    previous_loss_d = float('inf')
    fid_update_interval = 10  # Calculate FID every 10 batches

    # Create a log file
    log_file = os.path.join(OUTPUT_DIR, 'training_log.txt')
    with open(log_file, 'w') as f:
        f.write('Epoch,Batch,Loss_D,Loss_G,FID,Reward,LR,Beta1\n')

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            try:
                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # Train with real
                netD.zero_grad()
                real_data = data[0].to(device)
                batch_size = real_data.size(0)
                real_label = torch.full((batch_size,), 1, dtype=torch.float, device=device)
                fake_label = torch.full((batch_size,), 0, dtype=torch.float, device=device)

                # Forward pass for discriminator on real data
                output = netD(real_data)
                # Ensure proper dimensions
                output = output.view(-1)
                errD_real = criterion(output, real_label)
                errD_real.backward()

                # Train with fake
                noise = torch.randn(batch_size, 100, 1, 1, device=device)
                fake_data = netG(noise)
                output = netD(fake_data.detach())
                # Ensure proper dimensions
                output = output.view(-1)
                errD_fake = criterion(output, fake_label)
                errD_fake.backward()
                errD = errD_real.item() + errD_fake.item()
                
                # Update discriminator
                optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                netG.zero_grad()
                output = netD(fake_data)
                # Ensure proper dimensions
                output = output.view(-1)
                errG = criterion(output, real_label)
                errG.backward()
                optimizerG.step()
                
                # Calculate rewards and update hyperparameters occasionally
                if i % 50 == 0:
                    try:
                        # Calculate FID score periodically
                        if i % fid_update_interval == 0:
                            with torch.no_grad():
                                current_fid = fid_calculator.calculate_fid(real_data, fake_data)
                        else:
                            current_fid = previous_fid  # Use previous FID score
                        
                        # Calculate diversity score
                        with torch.no_grad():
                            diversity_score = calculate_diversity_score(fake_data)
                        
                        # Handle NaN or infinite values
                        if np.isnan(current_fid) or np.isinf(current_fid):
                            current_fid = previous_fid
                        if np.isnan(diversity_score) or np.isinf(diversity_score):
                            diversity_score = 0.0
                        
                        # Calculate reward
                        reward = calculate_rewards(
                            epoch, num_epochs, current_fid, previous_fid,
                            errG.item(), errD, diversity_score
                        )
                        
                        # Handle NaN reward
                        if np.isnan(reward) or np.isinf(reward):
                            reward = 0.0
                        
                        # Prepare state for RL agent
                        state = np.array([
                            [epoch / num_epochs,
                             min(current_fid / 100.0, 1.0),  # Normalize and clip FID score
                             min(errG.item(), 10.0),  # Clip generator loss
                             min(errD, 10.0)]  # Clip discriminator loss
                        ])
                        
                        # Get new hyperparameters from RL agent
                        new_params, log_prob = rl_agent.get_action(state)
                        
                        # Store transition
                        rl_agent.store_transition(state, new_params, reward, log_prob)
                        
                        # Train RL agent
                        rl_agent.train()
                        
                        # Update optimizers with new hyperparameters
                        for param_group in optimizerD.param_groups:
                            param_group['lr'] = new_params[0]
                            param_group['betas'] = (new_params[1], 0.999)
                        
                        for param_group in optimizerG.param_groups:
                            param_group['lr'] = new_params[0]
                            param_group['betas'] = (new_params[1], 0.999)
                        
                        # Update current_params for logging
                        current_params['lr'] = new_params[0]
                        current_params['beta1'] = new_params[1]
                        
                        # Update previous values
                        previous_fid = current_fid
                        previous_loss_g = errG.item()
                        previous_loss_d = errD
                        
                        # Print and log training information
                        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f FID: %.4f Reward: %.4f'
                              % (epoch, num_epochs, i, len(dataloader), errD, errG.item(), current_fid, reward))
                        
                        # Log to file
                        with open(log_file, 'a') as f:
                            f.write(f'{epoch},{i},{errD:.4f},{errG.item():.4f},{current_fid:.4f},{reward:.4f},'
                                   f'{current_params["lr"]:.6f},{current_params["beta1"]:.3f}\n')
                        
                        # Save generated images occasionally
                        if i % 100 == 0:
                            with torch.no_grad():
                                fake_images = netG(fixed_noise)
                                vutils.save_image(fake_images.detach(),
                                            f'{OUTPUT_DIR}/fake_samples_epoch_{epoch}_batch_{i}.png',
                                            normalize=True)
                    except Exception as e:
                        print(f"Error in reward calculation or hyperparameter update: {str(e)}")
                        continue
                        
            except Exception as e:
                print(f"Error in training loop: {str(e)}")
                continue

    # Save final model
    torch.save(netG.state_dict(), os.path.join(OUTPUT_DIR, 'generator.pth'))
    torch.save(netD.state_dict(), os.path.join(OUTPUT_DIR, 'discriminator.pth'))
    
    # Create training progress video
    create_training_video(OUTPUT_DIR)
    
    # Plot training progress
    plot_training_progress(log_file)
    
    print("Training complete!")

if __name__ == "__main__":
    train_dcgan() 