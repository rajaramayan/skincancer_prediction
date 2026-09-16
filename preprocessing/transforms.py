"""
Image preprocessing and augmentation transforms for HAM10000 dermoscopy images
"""

from typing import Dict, Any
import torch
import torchvision.transforms as transforms

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    # Fallback to torchvision transforms
    print("Warning: albumentations not available. Using torchvision transforms.")


def get_transforms(augmentation_level: str = "medium", image_size: int = 224) -> Dict[str, Any]:
    """
    Get augmentation transforms for training and validation (HAM10000)
    
    Args:
        augmentation_level: "light", "medium", or "heavy"
        image_size: Size to resize images to
        
    Returns:
        Dictionary with train_transform and val_transform
    """
    
    if ALBUMENTATIONS_AVAILABLE:
        return _get_albumentations_transforms(augmentation_level, image_size)
    else:
        return _get_torchvision_transforms(augmentation_level, image_size)


def _get_albumentations_transforms(augmentation_level: str = "medium", image_size: int = 224):
    """Get albumentations transforms"""
    # Normalization for ImageNet pretrained models
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Validation transform (minimal augmentation)
    val_transform = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    # Training transforms based on augmentation level
    if augmentation_level == "light":
        train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    elif augmentation_level == "medium":
        train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.GaussNoise(p=0.2),
            A.GaussBlur(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    else:  # heavy
        train_transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=40, p=0.7),
            A.Perspective(scale=(0.05, 0.1), p=0.3),
            A.ElasticTransform(p=0.3),
            A.GaussNoise(p=0.3),
            A.GaussBlur(p=0.3),
            A.RandomBrightnessContrast(p=0.4),
            A.RandomRain(p=0.1),
            A.RandomFog(p=0.1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    
    return {
        'train_transform': train_transform,
        'val_transform': val_transform
    }


def _get_torchvision_transforms(augmentation_level: str = "medium", image_size: int = 224):
    """Fallback to torchvision transforms when albumentations is not available"""
    
    # Normalization for ImageNet pretrained models
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=mean, std=std)
    
    # Validation transform
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Training transforms
    if augmentation_level == "light":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            normalize
        ])
    elif augmentation_level == "medium":
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:  # heavy
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(40),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=30, scale=(0.8, 1.2)),
            transforms.ToTensor(),
            normalize
        ])
    
    return {
        'train_transform': train_transform,
        'val_transform': val_transform
    }


def get_weak_augmentation(image_size: int = 224):
    """
    Weak augmentation for consistency regularization in FixMatch
    Used for generating weakly augmented versions of unlabeled data
    """
    if ALBUMENTATIONS_AVAILABLE:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=15, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])


def get_strong_augmentation(image_size: int = 224):
    """
    Strong augmentation for FixMatch
    Applies stronger transformations to unlabeled data while maintaining label correctness
    Uses RandAugment-style augmentation
    """
    if ALBUMENTATIONS_AVAILABLE:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        return A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.6),
            A.Perspective(scale=(0.05, 0.15), p=0.4),
            A.ElasticTransform(p=0.4),
            A.GaussNoise(p=0.4),
            A.GaussBlur(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.RandomRain(p=0.15),
            A.RandomFog(p=0.15),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomAffine(degrees=40, scale=(0.7, 1.3)),
            transforms.ToTensor(),
            normalize
        ])


def get_test_transform(image_size: int = 224):
    """
    Transforms for inference/testing (no augmentation, only normalization)
    """
    if ALBUMENTATIONS_AVAILABLE:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        
        return A.Compose([
            A.Resize(image_size, image_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean, std=std)
        
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize
        ])
