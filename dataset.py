import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class LFWDataset(Dataset):   
    def __init__(self, pairs_file, img_folder, transform=None):
        """
        Args:
            pairs_file (str): Path to the pairs file
            img_folder (str): Path to the LFW images folder
            transform (callable, optional): Optional transform to be applied on the images
        """
        self.pairs_file = pairs_file
        self.img_folder = img_folder
        self.transform = transform
        self.image_pairs, self.labels = self._load_pairs()

    def _load_pairs(self):
        """Load image pairs and labels from the pairs file"""
        with open(self.pairs_file, 'r') as f:
            pairs = f.readlines()[1:]  # Skip the header line
        
        image_pairs = []
        labels = []

        for pair in pairs:
            pair = pair.strip().split()
            if len(pair) == 3:  # Same person, different images
                person = pair[0]
                img_1 = os.path.join(self.img_folder, person, f"{person}_{int(pair[1]):04d}.jpg")
                img_2 = os.path.join(self.img_folder, person, f"{person}_{int(pair[2]):04d}.jpg")
                label = 1  # Same identity
            else:  # Different people
                person_1 = pair[0]
                person_2 = pair[2]
                img_1 = os.path.join(self.img_folder, person_1, f"{person_1}_{int(pair[1]):04d}.jpg")
                img_2 = os.path.join(self.img_folder, person_2, f"{person_2}_{int(pair[3]):04d}.jpg")
                label = 0  # Different identity

            # Check if both images exist
            if os.path.exists(img_1) and os.path.exists(img_2):
                image_pairs.append((img_1, img_2))
                labels.append(label)

        return image_pairs, labels

    def __getitem__(self, index):
        """Get a pair of images and their similarity label"""
        path_img_1, path_img_2 = self.image_pairs[index]
        label = self.labels[index]

        img_1 = Image.open(path_img_1).convert("RGB")
        img_2 = Image.open(path_img_2).convert("RGB")

        if self.transform:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, img_2, torch.tensor([label], dtype=torch.float32)

    def __len__(self):
        """Return the number of image pairs"""
        return len(self.image_pairs)


def get_transforms(image_size=224):
    """
    Create image transformations
    
    Args:
        image_size (int): Image size for resizing
        
    Returns:
        transforms.Compose: Composed transforms
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])


def get_dataloaders(config):
    """
    Create train and validation dataloaders
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    transform = get_transforms(config['dataset']['image_size']) 
    # Create dataset
    dataset = LFWDataset(
        pairs_file=config['dataset']['pairs_file'],
        img_folder=config['dataset']['img_folder'],
        transform=transform
    )
    
    # Split dataset
    train_size = int(config['dataset']['train_ratio'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=config['dataloader']['shuffle'],
        num_workers=config['dataloader']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['dataloader']['batch_size'],
        shuffle=False,
        num_workers=config['dataloader']['num_workers']
    )
    
    return train_loader, val_loader