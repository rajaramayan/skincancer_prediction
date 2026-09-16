"""
Model architectures for skin cancer classification
"""

from .backbone import create_backbone
from .supervised import SupervisedModel
from .fixmatch import FixMatchModel

__all__ = ["create_backbone", "SupervisedModel", "FixMatchModel"]
