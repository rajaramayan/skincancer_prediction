"""
Preprocessing module for data loading and augmentation
"""

# Lazy imports to avoid import errors during module initialization
__all__ = ["SkinCancerDataset", "HAM10000Dataset", "get_transforms", "split_dataset", "create_split_files"]

def __getattr__(name):
    if name == "SkinCancerDataset":
        from .dataset import SkinCancerDataset
        return SkinCancerDataset
    elif name == "HAM10000Dataset":
        from .dataset import HAM10000Dataset
        return HAM10000Dataset
    elif name == "get_transforms":
        from .transforms import get_transforms
        return get_transforms
    elif name == "split_dataset":
        from .split_data import split_dataset
        return split_dataset
    elif name == "create_split_files":
        from .split_data import create_split_files
        return create_split_files
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
