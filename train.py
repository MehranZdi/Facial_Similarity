import os
import torch
import torch.optim as optim
from tqdm import tqdm
from model import SiameseNetwork, ContrastiveLoss
from dataset import get_dataloaders
from utils import save_checkpoint, load_checkpoint, AverageMeter


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch
    
    Args:
        model: Siamese network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use (cuda/cpu)
        
    Returns:
        float: Average loss for the epoch
    """
    model.train()
    losses = AverageMeter('Loss')
    
    train_loop = tqdm(train_loader, desc="Training", leave=False)
    
    for img_1, img_2, label in train_loop:
        # Move tensors to device
        img_1, img_2, label = img_1.to(device), img_2.to(device), label.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output_1, output_2 = model(img_1, img_2)
        loss = criterion(output_1, output_2, label)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update statistics
        losses.update(loss.item(), img_1.size(0))
        train_loop.set_postfix(loss=losses.avg)
    
    return losses.avg


def validate(model, val_loader, criterion, device):
    """
    Validate the model
    
    Args:
        model: Siamese network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to use (cuda/cpu)
        
    Returns:
        float: Average validation loss
    """
    model.eval()
    losses = AverageMeter('Loss')
    
    val_loop = tqdm(val_loader, desc="Validation", leave=False)
    
    with torch.no_grad():
        for img_1, img_2, label in val_loop:
            # Move tensors to device
            img_1, img_2, label = img_1.to(device), img_2.to(device), label.to(device)
            
            # Forward pass
            output_1, output_2 = model(img_1, img_2)
            loss = criterion(output_1, output_2, label)
            
            # Update statistics
            losses.update(loss.item(), img_1.size(0))
            val_loop.set_postfix(loss=losses.avg)
    
    return losses.avg


def train_model(config):
    """
    Train the Siamese network
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        model: Trained model
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders
    train_loader, val_loader = get_dataloaders(config)
    print(f"Training set size: {len(train_loader.dataset)}")
    print(f"Validation set size: {len(val_loader.dataset)}")
    
    # Create model
    model = SiameseNetwork(
        backbone=config['model']['backbone'],
        pretrained=config['model']['pretrained'],
        embedding_size=config['model']['embedding_size'],
        freeze_backbone=config['model']['freeze_backbone']
    ).to(device)
    
    # Create loss function and optimizer
    criterion = ContrastiveLoss(margin=config['training']['margin'])
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['learning_rate']
    )
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_val_loss = float('inf')
    start_epoch = 0
    
    # Check if checkpoint exists
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(checkpoint_path):
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, checkpoint_path)
    
    epochs = config['training']['epochs']
    for epoch in range(start_epoch, epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")
        
        # Train and validate
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        # Print epoch results
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save checkpoint if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving model checkpoint (Val Loss: {best_val_loss:.4f})")
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss
            }, os.path.join(checkpoint_dir, 'best_model.pth'))
    
    return model