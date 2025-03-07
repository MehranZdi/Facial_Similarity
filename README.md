# Facial Similarity with Siamese Networks

This project implements a facial similarity detection system using Siamese Networks with a pretrained ResNet backbone. The system can determine whether two face images belong to the same person by learning a similarity metric between face embeddings.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Training Process](#training-process)
- [Results Visualization](#results-visualization)
- [Contributing](#contributing)
- [License](#license)
  


## Overview

Facial similarity detection is a common task in computer vision with applications in security, user authentication, and photo organization. This project uses a Siamese Network architecture, which consists of two identical neural networks that share weights. The network is trained to generate embeddings for face images such that images of the same person have embeddings close together in the embedding space, while images of different people have embeddings far apart.

The project uses the Labeled Faces in the Wild (LFW) dataset, which contains more than 13,000 face images of various celebrities and public figures collected from the web.

## Features

- **Pretrained ResNet backbone**: Leverages transfer learning with ResNet18, ResNet34, or ResNet50
- **Fine-tuning capabilities**: Option to freeze or unfreeze the backbone layers
- **Contrastive loss function**: Optimized for face similarity learning
- **Modular architecture**: Well-organized code structure for easy modification and extension
- **Comprehensive configuration**: YAML-based configuration for all parameters
- **Checkpointing**: Save and restore model states for continued training
- **Visualization tools**: Functions to visualize similarity between face pairs
- **Command-line interface**: Easy training and testing from the command line

## Project Structure

```
Facial_Similarity/
├── config.yaml          # Configuration parameters
├── dataset.py           # Dataset handling code
├── model.py             # Model architecture
├── train.py             # Training loop
├── utils.py             # Utility functions
├── main.py              # Entry point script
└── README.md            # This documentation file
```

## Installation

### Prerequisites

- torch and cuda
- torchvision
- PIL (Pillow)
- matplotlib
- tqdm
- PyYAML
- argparse
- numpy
- kagglehub

### Setting up the environment

```bash
# Create a virtual environment
python -m venv env

# Activate the environment
# On Windows
env\Scripts\activate
# On macOS/Linux
source env/bin/activate

# Install dependencies using:
pip install -r requirements.txt
```

## Dataset

This project uses the Labeled Faces in the Wild (LFW) dataset.Kagglehub has been used to download the dataset.

The LFW dataset comes with pairs files that specify which face pairs should be compared. These files are in the format:

```
10000
George_W_Bush 1 4
George_W_Bush 5 8
...
```

Where the first line specifies the number of pairs, and each subsequent line specifies a pair. For same-person pairs, the format is `<name> <image_number1> <image_number2>`. For different-person pairs, the format is `<name1> <image_number1> <name2> <image_number2>`.

## Configuration

All parameters are specified in `config.yaml`. Here's an explanation of the key configuration options:

```yaml
# Dataset Configuration
dataset:
  pairs_file: "path/to/pairsDevTrain.txt"  # Path to pairs file
  img_folder: "path/to/lfw-deepfunneled"    # Path to image folder
  train_ratio: 0.8                          # Train/validation split ratio
  image_size: 224                           # Input image size (ResNet default)

# DataLoader Configuration
dataloader:
  batch_size: 32                            # Batch size for training
  num_workers: 4                            # Number of data loading workers
  shuffle: true                             # Whether to shuffle the training data

# Model Configuration
model:
  backbone: "resnet18"                      # Backbone model (resnet18, resnet34, resnet50)
  pretrained: true                          # Whether to use pretrained weights
  embedding_size: 128                       # Size of the face embedding vector
  freeze_backbone: false                    # Whether to freeze backbone weights

# Training Configuration
training:
  epochs: 20                                # Number of training epochs
  learning_rate: 0.001                      # Learning rate for Adam optimizer
  margin: 1.0                               # Margin for contrastive loss
  checkpoint_dir: "checkpoints"             # Directory to save checkpoints
  save_best_only: true                      # Save only the best model
```

## Usage

### Training the model

```bash
python main.py --train
```

### Testing the model on a pair of images

```bash
python main.py --test --img1 path/to/image1.jpg --img2 path/to/image2.jpg
```

### Both training and testing

```bash
python main.py --train --test --img1 path/to/image1.jpg --img2 path/to/image2.jpg
```

## Architecture

### Siamese Network

The Siamese Network consists of:

1. **Backbone**: A pretrained ResNet model (18, 34, or 50) with the final classification layer removed.
2. **Embedding Network**: Additional layers added on top of the backbone to create a fixed-size embedding vector:
   - Flatten layer
   - Linear layer (512 units)
   - ReLU activation
   - Batch Normalization
   - Dropout (0.3)
   - Linear layer (to embedding_size, default 128)
   - L2 Normalization

### Contrastive Loss

The model is trained using contrastive loss, which:
- Minimizes the distance between embeddings of the same person
- Ensures that embeddings of different people are separated by at least a margin

The loss function is defined as:
```
L(E1, E2, Y) = (1-Y) * d(E1, E2)² + Y * max(0, margin - d(E1, E2))²
```
where:
- E1, E2 are the embeddings
- Y is 0 for same person, 1 for different people
- d(E1, E2) is the Euclidean distance between embeddings

## Training Process

1. **Data Preparation**:
   - Load image pairs from the pairs file
   - Apply transformations (resize, normalize)
   - Split into training and validation sets

2. **Training Loop**:
   - Forward pass pairs through the Siamese network
   - Compute contrastive loss
   - Backpropagate and update weights
   - Evaluate on validation set
   - Save checkpoint if validation loss improves

3. **Evaluation**:
   - Compute embeddings for two input faces
   - Calculate Euclidean distance between embeddings
   - Convert to similarity score (closer to 1 means more similar)
   - Classify as same person if similarity > threshold (typically 0.5)

## Results Visualization

The `show_result` function in `utils.py` visualizes the similarity between two face images:

- Displays the two images side by side
- Shows the similarity score and Euclidean distance
- Indicates whether the system considers them the same person


## Contributing

Feel free to fork this repository and submit pull requests!

## License

This project is licensed under the [MIT](https://opensource.org/license/mit) License.

