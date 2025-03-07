import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_config(config_path):
    """
    Load configuration from YAML file
    
    Args:
        config_path (str): Path to the configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_checkpoint(state, filename):
    """
    Save model checkpoint
    
    Args:
        state (dict): State dictionary containing model parameters
        filename (str): Path to save the checkpoint
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(model, optimizer, filename):
    """
    Load model checkpoint
    
    Args:
        model: Model to load the parameters into
        optimizer: Optimizer to load the parameters into (can be None)
        filename (str): Path to the checkpoint file
        
    Returns:
        tuple: (epoch, loss)
    """
    if not os.path.exists(filename):
        print(f"No checkpoint found at {filename}")
        return 0, float('inf')
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Only load optimizer state if optimizer is provided
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded: Epoch {epoch}, Loss {loss:.4f}")
    return epoch, loss


class AverageMeter:
    """
    Computes and stores the average and current value
    """
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def show_result(model, img1_path, img2_path, transform, device=None):
    """
    Show similarity result for two images
    
    Args:
        model: Trained Siamese network
        img1_path (str): Path to first image
        img2_path (str): Path to second image
        transform: Image transformations
        device: Device to use (cuda/cpu)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and transform images
    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")
    
    img1_tensor = transform(img1).unsqueeze(0).to(device)
    img2_tensor = transform(img2).unsqueeze(0).to(device)
    
    # Get similarity score
    model.eval()
    with torch.no_grad():
        output1, output2 = model(img1_tensor, img2_tensor)
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2).item()
    
    # Convert similarity score to a percentage (closer to 1 means more similar)
    similarity_score = max(0, 1 - euclidean_distance)
    
    # Display images and similarity score
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(np.array(img1))
    ax[0].axis('off')
    ax[0].set_title('Image 1')
    
    ax[1].imshow(np.array(img2))
    ax[1].axis('off')
    ax[1].set_title('Image 2')
    
    plt.suptitle(f'Similarity Score: {similarity_score:.2f} (Distance: {euclidean_distance:.2f})', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return similarity_score, euclidean_distance