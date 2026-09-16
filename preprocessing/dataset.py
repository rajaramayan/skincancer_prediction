"""
Dataset management for skin cancer classification (HAM10000)
Handles labeled and unlabeled data loading from HAM10000 dataset
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


def _apply_transform(image: Image.Image, transform) -> torch.Tensor:
    """Apply an albumentations or torchvision transform to a PIL image"""
    if transform:
        try:
            # Try albumentations first (expects numpy array)
            image_np = np.array(image)
            result = transform(image=image_np)
            return result['image']
        except (TypeError, KeyError):
            # Fall back to torchvision (expects PIL Image)
            return transform(image)
    # Convert to tensor if no transforms
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0


class HAM10000Dataset(Dataset):
    """
    Dataset class for HAM10000 skin cancer images
    Supports both labeled and unlabeled data
    
    HAM10000 classes:
    - nv: Nevus (benign)
    - mel: Melanoma (malignant)
    - bkl: Benign keratosis-like lesions
    - bcc: Basal cell carcinoma
    - akiec: Actinic keratosis / Bowen's disease
    - vasc: Vascular lesions
    - df: Dermatofibroma
    """
    
    # Class to index mapping
    CLASS_TO_IDX = {
        'nv': 0,      # Nevus
        'mel': 1,     # Melanoma
        'bkl': 2,     # Benign keratosis
        'bcc': 3,     # Basal cell carcinoma
        'akiec': 4,   # Actinic keratosis
        'vasc': 5,    # Vascular
        'df': 6       # Dermatofibroma
    }
    
    IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}
    
    def __init__(
        self,
        image_dir: str,
        metadata_file: str,
        transform=None,
        unlabeled: bool = False,
        image_indices: Optional[List[int]] = None
    ):
        """
        Args:
            image_dir: Directory containing HAM10000 images
            metadata_file: CSV file with image metadata (hmnist_20_metadata.csv)
            transform: Torchvision transforms to apply
            unlabeled: If True, treats data as unlabeled (no labels returned)
            image_indices: Specific indices to use (for train/val splits)
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.unlabeled = unlabeled
        
        # Load metadata
        self.metadata = pd.read_csv(metadata_file)
        
        # Use specific indices if provided, otherwise use all
        if image_indices is not None:
            self.metadata = self.metadata.iloc[image_indices].reset_index(drop=True)
        else:
            self.metadata = self.metadata.reset_index(drop=True)
        
        # Get image paths and labels
        self.images = []
        self.labels = []
        
        for idx, row in self.metadata.iterrows():
            image_id = row['image_id']
            # Handle both .jpg and .png extensions
            img_path = self.image_dir / f"{image_id}.jpg"
            if not img_path.exists():
                img_path = self.image_dir / f"{image_id}.png"
            
            if img_path.exists():
                self.images.append(str(img_path))
                # Map diagnosis to class index
                dx = row['dx']
                self.labels.append(self.CLASS_TO_IDX.get(dx, -1))
        
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get item by index
        
        Returns:
            Tuple of (image, label) or (image,) if unlabeled
        """
        # Load image
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        image = _apply_transform(image, self.transform)
        
        if self.unlabeled:
            return image
        else:
            label = self.labels[idx]
            return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes in the dataset"""
        dist = {}
        for label in self.labels:
            class_name = self.IDX_TO_CLASS[label]
            dist[class_name] = dist.get(class_name, 0) + 1
        return dist
    
    def get_metadata(self, idx: int) -> Dict:
        """Get metadata for a sample (age, sex, localization)"""
        row = self.metadata.iloc[idx]
        return {
            'image_id': row['image_id'],
            'dx': row['dx'],
            'age': row.get('age', None),
            'sex': row.get('sex', None),
            'localization': row.get('localization', None)
        }


class SkinCancerDataset(HAM10000Dataset):
    """Alias for backward compatibility"""
    pass


class FixMatchUnlabeledDataset(Dataset):
    """
    Unlabeled dataset for FixMatch that returns a (weakly-augmented, strongly-augmented)
    pair of views for each image, as required for consistency regularization
    """

    def __init__(
        self,
        image_dir: str,
        metadata_file: str,
        weak_transform,
        strong_transform,
        image_indices: Optional[List[int]] = None
    ):
        self.image_dir = Path(image_dir)
        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

        metadata = pd.read_csv(metadata_file)
        if image_indices is not None:
            metadata = metadata.iloc[image_indices].reset_index(drop=True)
        else:
            metadata = metadata.reset_index(drop=True)

        self.images = []
        for _, row in metadata.iterrows():
            image_id = row['image_id']
            img_path = self.image_dir / f"{image_id}.jpg"
            if not img_path.exists():
                img_path = self.image_dir / f"{image_id}.png"
            if img_path.exists():
                self.images.append(str(img_path))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple:
        image = Image.open(self.images[idx]).convert('RGB')
        weak_view = _apply_transform(image, self.weak_transform)
        strong_view = _apply_transform(image, self.strong_transform)
        return weak_view, strong_view
