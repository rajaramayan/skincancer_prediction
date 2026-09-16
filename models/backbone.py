"""
Backbone models for feature extraction
"""

from typing import Optional
import torch
import torch.nn as nn
import torchvision.models as models


def create_backbone(
    model_name: str = "resnet50",
    pretrained: bool = True,
    num_classes: int = 7
):
    """
    Create a backbone model
    
    Args:
        model_name: Name of the model (resnet50, resnet18, efficientnet, etc.)
        pretrained: Whether to use pretrained weights
        num_classes: Number of output classes
        
    Returns:
        PyTorch model
    """
    
    if model_name.lower() == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        # Replace final layer
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    elif model_name.lower() == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    elif model_name.lower() == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    
    elif model_name.lower() == "densenet":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    
    elif model_name.lower() == "mobilenet":
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    
    else:
        # Default to ResNet50
        print(f"Model {model_name} not found. Using ResNet50.")
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
