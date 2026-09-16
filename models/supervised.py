"""
Supervised baseline model for skin cancer classification
"""

import torch
import torch.nn as nn


class SupervisedModel(nn.Module):
    """
    Supervised baseline model trained on labeled data only
    """
    
    def __init__(self, backbone, num_classes: int = 7):
        """
        Args:
            backbone: Feature extraction backbone
            num_classes: Number of output classes
        """
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
    def forward(self, x):
        """Forward pass"""
        return self.backbone(x)
    
    def get_embeddings(self, x):
        """Extract feature embeddings before final layer"""
        # For ResNet, get features from before the final FC layer
        if hasattr(self.backbone, 'avgpool'):
            features = self.backbone.avgpool(self.backbone.layer4(self.backbone.layer3(
                self.backbone.layer2(self.backbone.layer1(self.backbone.relu(
                    self.backbone.bn1(self.backbone.conv1(x))))))))
            features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            return features.view(features.size(0), -1)
        else:
            # Generic fallback
            return x
