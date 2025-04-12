# DCGAN with RL-Guided Hyperparameter Optimization

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) enhanced with a Reinforcement Learning (RL) agent to dynamically adjust critical hyperparameters during training. The integrated RL component employs a Proximal Policy Optimization (PPO) approach for fine-tuning the learning rate and momentum (beta1) in response to evolving training metrics.
Due to the limited resources, only fewer epochs are tested. You can see the process how this model trained by visualized video.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Components](#key-components)
  - [DCGAN Architecture](#dcgan-architecture)
  - [RL Agent: PPOHyperparameterOptimizer](#rl-agent-ppohyperparameteroptimizer)
- [Training Features](#training-features)
- [Main Functions](#main-functions)
- [Usage Instructions](#usage-instructions)
- [Training Parameters](#training-parameters)
- [Output Files](#output-files)
- [Dependencies](#dependencies)
- [Additional Notes](#additional-notes)

---

## Overview

This repository demonstrates a novel approach to training generative models by combining the adversarial framework of DCGAN with reinforcement learning to optimize training hyperparameters dynamically. The system is designed to generate realistic 64x64 RGB images from the CelebA dataset while continuously refining its training regimen to improve results.

---

## Project Structure

- **`dcgan_basic.py`**  
  The main script that:
  - Loads and preprocesses the CelebA dataset.
  - Defines the DCGAN generator and discriminator architectures.
  - Incorporates an RL agent to update training hyperparameters.
  - Implements training loops with periodic logging, visualization, and video generation.

- **Supporting Modules and Functions**  
  - Reward and diversity score calculation functions.
  - Utility functions for model weight initialization, image visualization, and progress logging.

---

## Key Components

### DCGAN Architecture

- **Generator**
  - **Input:** A 100-dimensional noise vector.
  - **Transformation:** A series of transposed convolutional layers progressively upsample the noise into a 64x64 RGB image.
  - **Activation:** Uses ReLU for hidden layers and Tanh for the output, ensuring generated pixel values are in the range [-1, 1].

- **Discriminator**
  - **Input:** A 64x64 RGB image.
  - **Transformation:** Convolutional layers with LeakyReLU activations and Batch Normalization extract image features.
  - **Output:** A scalar value indicating whether the input image is real or fake.

- **Training Process**
  - **Adversarial Training:** Alternating optimization of the generator and discriminator.
  - **Loss Function:** Uses BCEWithLogitsLoss for stability in both discriminator and generator training.

### RL Agent: PPOHyperparameterOptimizer

- **Purpose:**  
  Optimizes hyperparameters (learning rate and beta1) during training, adapting to the current performance of the model.

- **State Space:**  
  Composed of:
  - **Epoch Progress:** Normalized progress of the training epoch.
  - **FID Score:** A metric of image quality (simplified calculation in this implementation).
  - **Generator Loss:** Current loss value for the generator.
  - **Discriminator Loss:** Current loss value for the discriminator.

- **Action Space:**  
  Determines two hyperparameters:
  - **Learning Rate:** Adjusted to roughly fall in the range of 0.0001 to 0.0003.
  - **Beta1:** Momentum term for the Adam optimizer, scaled between 0.3 and 0.7.

- **Reward Function:**  
  A composite metric combining:
  - **FID Improvement:** Change in FID score between epochs.
  - **Loss Balance:** Difference between generator and discriminator loss (preventing extreme divergence).
  - **Diversity Score:** Variance-based metric to promote image diversity.

---

## Training Features

- **Mixed Precision Training:**  
  Utilizes PyTorch's AMP (Automatic Mixed Precision) for performance improvements when using GPU acceleration.

- **Dynamic Hyperparameter Adjustment:**  
  The RL agent updates optimizer parameters every 50 training batches, allowing for adaptive learning based on current progress.

- **Visualization and Logging:**  
  Detailed logging of training metrics (losses, FID, rewards, and hyperparameters) with functionalities to:
  - Save generated image samples periodically.
  - Generate training progress plots.
  - Create a time-lapse video capturing the evolution of generated images.

---

## Main Functions

1. **`train_dcgan()`**  
   - Orchestrates the entire training process.
   - Initializes models, data loaders, and the RL agent.
   - Runs the training loop over a specified number of epochs.
   - Integrates periodic saving of generated images and model weights.

2. **`calculate_rewards()`**  
   - Computes a reward based on the current FID, loss values, and a diversity score.
   - Incorporates a dynamic weight mechanism dependent on training progress.

3. **`calculate_diversity_score()`**  
   - Measures the diversity of generated images using the variance across a batch.

4. **`create_training_video()`**  
   - Compiles saved image samples into a video to visualize training progression.

5. **`plot_training_progress()`**  
   - Reads log files and plots metrics such as losses, FID, and rewards.

---

## Usage Instructions

1. **Prepare the Environment:**  
   Ensure the CelebA dataset is downloaded and placed in the specified directory.

2. **Run the Script:**  
   Execute the training process via the command line:
   ```bash
   python dcgan_basic.py
3. **Training Process Flow:**

-The script initializes the DCGAN models and the RL agent.

-The network trains for 50 epochs, periodically saving generated image samples.

-Progress is logged to a file and visualized through plots and a training video.

-Final model weights for both the generator and discriminator are saved upon completion.

---
## Dependencies

Ensure the following libraries are installed:

- **PyTorch**
- **torchvision**
- **numpy**
- **matplotlib**
- **moviepy**
- **PIL (Python Imaging Library)**

You can install these dependencies via pip using the command below:

```bash
pip install torch torchvision numpy matplotlib moviepy pillow
```
---

## Training Parameters

-Batch Size: 32

-Image Dimensions: 64x64 RGB

-Number of Epochs: 50

-Initial Learning Rate: 0.0002

-Initial Beta1: 0.5

-FID Update Interval: Every 10 batches

-RL Update Frequency: Every 50 batches

---
## Output Files

After training, the following files will be generated:

- **generator.pth**: Saved weights for the trained generator model.
- **discriminator.pth**: Saved weights for the trained discriminator model.
- **training_progress.mp4**: A video summarizing the training process.
- **training_progress.png**: Plots of training metrics (loss, FID, rewards).
- **Generated Image Samples**: Saved periodically during training for visual inspection.

---

## Additional Notes

### FID Calculation
- The FID (Fréchet Inception Distance) calculation is simplified in this implementation.
- For more robust performance assessment, consider integrating a more advanced FID computation.

### Hyperparameter Optimization
- The RL agent (PPOHyperparameterOptimizer) updates the learning rate and beta1 values dynamically.
- This approach is experimental and serves as a demonstration of how RL can be employed to fine-tune hyperparameters during the training process.

### Logging and Visualization
- Training progress is rigorously logged and visualized.
- Generated images, metrics plots, and a summary video provide insights into model performance over time.

### GPU Support
- The script automatically detects and uses GPU acceleration if available.
- Mixed precision training is employed to enhance performance on supported hardware.


