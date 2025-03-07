import argparse
import torch
from dataset import get_transforms
from model import SiameseNetwork
from train import train_model
from utils import load_config, show_result, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Facial Similarity with Siamese Networks')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model on image pair')
    parser.add_argument('--img1', type=str, help='Path to first test image')
    parser.add_argument('--img2', type=str, help='Path to second test image')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.train:
        # Train the model
        model = train_model(config)
        print("Training completed!")
    
    if args.test:
        if not args.img1 or not args.img2:
            print("Please provide paths to two images for testing")
            return
        
        # Load model
        model = SiameseNetwork(
            backbone=config['model']['backbone'],
            pretrained=False,  # We'll load weights from checkpoint
            embedding_size=config['model']['embedding_size']
        ).to(device)
        
        # Load checkpoint
        checkpoint_path = f"{config['training']['checkpoint_dir']}/best_model.pth"
        load_checkpoint(model, None, checkpoint_path)
        
        # Get transforms
        transform = get_transforms(config['dataset']['image_size'])
        
        # Show result
        similarity, distance = show_result(model, args.img1, args.img2, transform, device)
        print(f"Similarity: {similarity:.4f}, Distance: {distance:.4f}")
        print(f"Prediction: {'Same person' if similarity > 0.7 else 'Different people'}")


if __name__ == '__main__':
    main()