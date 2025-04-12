# DCGAN with RL-Guided Hyperparameter Optimization

This file implements a Deep Convolutional Generative Adversarial Network (DCGAN) with Reinforcement Learning (RL) based hyperparameter optimization.

## File Overview

`dcgan_basic.py` implements a DCGAN model that generates 64x64 RGB images from the CelebA dataset, with an integrated RL agent that dynamically optimizes training hyperparameters.

## Key Components

### 1. DCGAN Architecture
- **Generator**: Transforms 100-dimensional noise vectors into 64x64 RGB images
- **Discriminator**: Classifies images as real or fake
- **Training Process**: Alternating training of generator and discriminator

### 2. RL Agent (PPOHyperparameterOptimizer)
- **State Space**: [epoch progress, FID score, generator loss, discriminator loss]
- **Action Space**: [learning rate, beta1]
- **Reward Function**: Combines FID improvement, loss balance, and diversity metrics

### 3. Training Features
- Mixed precision training support
- Dynamic hyperparameter adjustment
- Progress visualization and logging
- Training video generation

## Main Functions

1. `train_dcgan()`: Main training function
2. `calculate_rewards()`: Computes reward for RL agent
3. `calculate_diversity_score()`: Measures image diversity
4. `create_training_video()`: Generates training progress video
5. `plot_training_progress()`: Visualizes training metrics

## Usage

```python
# Run training
python dcgan_basic.py
```

The script will:
1. Initialize DCGAN and RL agent
2. Train for 50 epochs
3. Save generated images periodically
4. Create training progress video
5. Save final model weights

## Training Parameters

- Batch size: 32
- Image size: 64x64
- Number of epochs: 50
- Initial learning rate: 0.0002
- Initial beta1: 0.5
- FID update interval: 10 batches

## Output Files

- `generator.pth`: Trained generator model weights
- `discriminator.pth`: Trained discriminator model weights
- `training_progress.mp4`: Video of training progress
- `training_progress.png`: Training metrics plots
- Generated image samples during training

## Dependencies

- PyTorch
- torchvision
- numpy
- matplotlib
- moviepy
- PIL

## Notes

- Uses simplified FID calculation for demonstration
- RL agent updates hyperparameters every 50 batches
- Training progress is logged and visualized
- Supports GPU acceleration when available 
