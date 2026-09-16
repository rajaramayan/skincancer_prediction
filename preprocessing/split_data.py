"""
Data splitting utilities for HAM10000 labeled/unlabeled data preparation
"""

from typing import Tuple, List, Dict
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


def split_dataset(
    metadata_file: str,
    labeled_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_seed: int = 42,
    stratify: bool = True
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split HAM10000 dataset into labeled, unlabeled, and validation sets
    Maintains class balance using stratified splitting
    
    Args:
        metadata_file: Path to HAM10000 metadata CSV file (hmnist_20_metadata.csv)
        labeled_ratio: Fraction of data to use as labeled (0.2 = 20%)
        val_ratio: Fraction of labeled data for validation (0.1 = 10% of labeled)
        random_seed: Random seed for reproducibility
        stratify: If True, maintain class distribution in splits
        
    Returns:
        Tuple of (labeled_indices, unlabeled_indices, val_indices)
    """
    
    # Load metadata
    metadata = pd.read_csv(metadata_file)
    all_indices = np.arange(len(metadata))
    
    # First split: labeled vs unlabeled
    if stratify:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=1 - labeled_ratio,
            random_state=random_seed
        )
        labeled_idx, unlabeled_idx = next(splitter.split(
            all_indices, 
            metadata['dx']
        ))
    else:
        labeled_idx, unlabeled_idx = train_test_split(
            all_indices,
            train_size=labeled_ratio,
            random_state=random_seed
        )
    
    # Second split: labeled into train and validation
    labeled_data = metadata.iloc[labeled_idx]
    
    if stratify:
        train_idx, val_idx = train_test_split(
            labeled_idx,
            test_size=val_ratio,
            random_state=random_seed,
            stratify=labeled_data['dx']
        )
    else:
        train_idx, val_idx = train_test_split(
            labeled_idx,
            test_size=val_ratio,
            random_state=random_seed
        )
    
    return train_idx.tolist(), val_idx.tolist(), unlabeled_idx.tolist()


def create_split_files(
    metadata_file: str,
    output_dir: str,
    labeled_ratio: float = 0.2,
    val_ratio: float = 0.1,
    random_seed: int = 42
) -> Dict[str, str]:
    """
    Create CSV files for train/val/unlabeled splits
    
    Args:
        metadata_file: Path to original metadata CSV
        output_dir: Directory to save split files
        labeled_ratio: Fraction for labeled data
        val_ratio: Fraction for validation
        random_seed: Random seed
        
    Returns:
        Dictionary with paths to created split files
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get splits
    train_idx, val_idx, unlabeled_idx = split_dataset(
        metadata_file,
        labeled_ratio=labeled_ratio,
        val_ratio=val_ratio,
        random_seed=random_seed
    )
    
    # Load full metadata
    metadata = pd.read_csv(metadata_file)
    
    # Create split files
    train_file = output_path / "train_split.csv"
    val_file = output_path / "val_split.csv"
    unlabeled_file = output_path / "unlabeled_split.csv"
    
    metadata.iloc[train_idx].to_csv(train_file, index=False)
    metadata.iloc[val_idx].to_csv(val_file, index=False)
    metadata.iloc[unlabeled_idx].to_csv(unlabeled_file, index=False)
    
    print(f"Train samples: {len(train_idx)}")
    print(f"Val samples: {len(val_idx)}")
    print(f"Unlabeled samples: {len(unlabeled_idx)}")
    
    # Print class distribution
    print("\nClass distribution in training set:")
    print(metadata.iloc[train_idx]['dx'].value_counts())
    
    return {
        'train': str(train_file),
        'val': str(val_file),
        'unlabeled': str(unlabeled_file)
    }


def get_class_distribution(metadata_file: str, indices: List[int] = None) -> Dict[str, int]:
    """
    Get class distribution for a subset of data
    
    Args:
        metadata_file: Path to metadata CSV
        indices: Specific indices to use (if None, uses all)
        
    Returns:
        Dictionary with class counts
    """
    metadata = pd.read_csv(metadata_file)
    
    if indices is not None:
        metadata = metadata.iloc[indices]
    
    return metadata['dx'].value_counts().to_dict()
