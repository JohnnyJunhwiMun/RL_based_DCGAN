# DCGAN with RL-Guided Hyperparameter Optimization

This project implements a Deep Convolutional Generative Adversarial Network (DCGAN) enhanced with a Reinforcement Learning (RL) agent that dynamically adjusts key hyperparameters during training. The RL component uses the Proximal Policy Optimization (PPO) algorithm to fine-tune the learning rate and momentum parameter (beta1) based on evolving training metrics.

Due to limited computational resources, the model was trained for only 10 epochs—fewer than typically used in similar projects. With better resource availability, there is significant potential for further improvements through extended training and more advanced tuning.

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
- [Dependencies](#dependencies)
- [Additional Notes and Results](#additional-notes-and-results)

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

This function computes the reward for the RL agent using a composite metric that balances improvements between epochs with the current performance level. It follows this formula:

```math
R_{\text{total}}(t) = \lambda(t) \cdot R_{\text{improve}}(t) + \bigl(1 - \lambda(t)\bigr) \cdot R_{\text{level}}(t)
```

Where:
- **$\(R_{\text{improve}}(t)\)$** evaluates the improvement from the previous state, combining:
  - **FID Improvement (50%)**: Difference in FID scores between epochs.
  - **Loss Balance (30%)**: Difference between generator and discriminator losses.
  - **Diversity (20%)**: Variance-based score to encourage image variety.

- **$\(R_{\text{level}}(t)\)$** assesses the current state by considering:
  - **FID Score (40%)**: Current FID value.
  - **Generator Loss (30%)**: Current generator loss.
  - **Diversity (30%)**: As above, to ensure image diversity.

- **$\(\lambda(t)\)$** is a dynamic weight that shifts during training:
  - **Early Stage (0–30%):** $\(\lambda(t) \approx 0.8\)$
  - **Middle Stage (30–70%):** $\(\lambda(t) \approx 0.5\)$
  - **Late Stage (70–100%):** $\(\lambda(t) \approx 0.3\)$

In short, `calculate_rewards()` blends these two reward components to provide a comprehensive signal that guides the RL agent in adjusting hyperparameters throughout the training process.

    
3. **`calculate_diversity_score()`**  
   - Measures the diversity of generated images using the variance across a batch.

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

## Additional Notes and Results

### FID Calculation
- The FID (Fréchet Inception Distance) calculation is simplified in this implementation.
- For more robust performance assessment, consider integrating a more advanced FID computation.

### Hyperparameter Optimization
- The RL agent (PPOHyperparameterOptimizer) updates the learning rate and beta1 values dynamically.
- This approach is experimental and serves as a demonstration of how RL can be employed to fine-tune hyperparameters during the training process.

### Logging and Visualization
- Training progress is rigorously logged and visualized.
- Generated images, metrics plots, and a summary video provide insights into model performance over time.
  
#### Visualizing training process
The visualization shows the average values per epoch, with an important detail after removing outliers using the IQR method.
<img src="https://github.com/user-attachments/assets/e5c828fd-1a40-46e7-93a9-f37f2e837a4f" alt="training_progress" width="700"/>

https://github.com/user-attachments/assets/66f45782-2eff-4731-b93b-74e52e14afaf

#### Improved images in final epoch (=10th epoch) in training process
![final_image](https://github.com/user-attachments/assets/52a0c6dc-b70e-4e9d-952c-76deaf1704d2)
#### Newly generated images
![generatedimage](https://github.com/user-attachments/assets/e036948c-e887-40c2-a7f6-b8a54443c8f8)



### GPU Support
- The script automatically detects and uses GPU acceleration if available.
- Mixed precision training is employed to enhance performance on supported hardware.


