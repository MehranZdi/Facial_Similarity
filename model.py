import torch
import torch.nn as nn
import torchvision.models as models


class SiameseNetwork(nn.Module):
    """
    Siamese Network using a pretrained ResNet18 backbone with additional layers
    for facial similarity recognition
    """
    
    def __init__(self, backbone='resnet18', pretrained=True, embedding_size=128, freeze_backbone=False):
        """
        Args:
            backbone (str): The backbone model to use ('resnet18' by default)
            pretrained (bool): Whether to use pretrained weights
            embedding_size (int): Size of the embedding vector
            freeze_backbone (bool): Whether to freeze the backbone weights
        """
        super(SiameseNetwork, self).__init__()
        
        # Load pretrained backbone
        if backbone == 'resnet18':
            base_model = models.resnet18(pretrained=pretrained)
            base_output_features = 512
        elif backbone == 'resnet34':
            base_model = models.resnet34(pretrained=pretrained)
            base_output_features = 512
        elif backbone == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            base_output_features = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Add custom embedding layers
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base_output_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_size)
        )
        
    def forward_once(self, x):
        """Forward pass for a single input"""
        x = self.backbone(x)
        x = self.embedding(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
    
    def forward(self, input_1, input_2):
        """Forward pass for a pair of inputs"""
        output_1 = self.forward_once(input_1)
        output_2 = self.forward_once(input_2)
        return output_1, output_2


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function for Siamese network
    """
    
    def __init__(self, margin=1.0):
        """
        Args:
            margin (float): Margin for dissimilar pairs
        """
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, output_1, output_2, label):
        """
        Compute contrastive loss
        
        Args:
            output_1: First embedding
            output_2: Second embedding
            label: 1 for similar pairs, 0 for dissimilar pairs
            
        Returns:
            loss: Scalar loss value
        """
        # Calculate the euclidean distance between embeddings
        euclidean_distance = nn.functional.pairwise_distance(output_1, output_2)
        
        # Calculate contrastive loss
        # For similar pairs (label=1): penalize distance
        # For dissimilar pairs (label=0): penalize if distance < margin
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +  # Similar pairs
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)  # Dissimilar pairs
        )
        
        return loss_contrastive