# ============================================================
# inference.py
# Skin Cancer Classification - ResNet18
# Trained using HAM10000 dataset
# ============================================================

import sys
from pathlib import Path

import torch
import numpy as np
from PIL import Image

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ============================================================
# 1. PROJECT PATH
# ============================================================

# Directory containing this inference.py file
PROJECT_DIR = Path(__file__).resolve().parent

# Add project directory to Python path
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


# ============================================================
# 2. IMPORT YOUR ORIGINAL MODEL CLASSES
# ============================================================

from models.backbone import create_backbone  # noqa: E402
from models.supervised import SupervisedModel  # noqa: E402


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

MODEL_PATH = PROJECT_DIR / "models" / "checkpoints" / "supervised_best.pt"

IMAGE_SIZE = 224

NUM_CLASSES = 7

BACKBONE_NAME = "resnet18"


# ============================================================
# 4. EXACT CLASS MAPPING USED DURING TRAINING
# ============================================================

# IMPORTANT:
# This is the exact mapping found in your
# preprocessing/dataset.py file.

CLASS_TO_IDX = {
    "nv": 0,       # Nevus
    "mel": 1,      # Melanoma
    "bkl": 2,      # Benign keratosis
    "bcc": 3,      # Basal cell carcinoma
    "akiec": 4,    # Actinic keratosis
    "vasc": 5,     # Vascular lesion
    "df": 6        # Dermatofibroma
}


# Reverse mapping
IDX_TO_CLASS = {
    value: key
    for key, value in CLASS_TO_IDX.items()
}


# Human-readable disease names
CLASS_LABELS = {
    "nv": "Nevus",
    "mel": "Melanoma",
    "bkl": "Benign Keratosis",
    "bcc": "Basal Cell Carcinoma",
    "akiec": "Actinic Keratosis",
    "vasc": "Vascular Lesion",
    "df": "Dermatofibroma"
}


# ============================================================
# 5. DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("SKIN CANCER MODEL - INFERENCE")
print("=" * 70)
print("Project directory:", PROJECT_DIR)
print("Model path:", MODEL_PATH)
print("Device:", DEVICE)
print("=" * 70)


# ============================================================
# 6. TEST / INFERENCE TRANSFORM
# ============================================================

# IMPORTANT:
# This must NOT contain random augmentation.
#
# The trained model expects:
# Resize → Normalize → Tensor
#
# This is consistent with the get_test_transform()
# function in your project.

test_transform = A.Compose([

    A.Resize(
        IMAGE_SIZE,
        IMAGE_SIZE
    ),

    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),

    ToTensorV2()

])


# ============================================================
# 7. CREATE MODEL
# ============================================================

def create_model():
    """
    Create the model architecture and load trained weights.

    Returns:
        model: PyTorch model in eval mode, moved to DEVICE
    """
    backbone = create_backbone(
        model_name=BACKBONE_NAME,
        pretrained=False,
        num_classes=NUM_CLASSES
    )

    model = SupervisedModel(backbone, num_classes=NUM_CLASSES)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# 8. PREDICTION
# ============================================================

def preprocess_image(image_source):
    """
    Load (if needed) and convert an image into a model-ready input tensor.

    Args:
        image_source: Path to the input image, or a PIL.Image instance

    Returns:
        (image, tensor): the RGB PIL.Image and a (1, C, H, W) tensor on DEVICE
    """
    if isinstance(image_source, Image.Image):
        image = image_source.convert("RGB")
    else:
        image = Image.open(image_source).convert("RGB")

    transformed = test_transform(image=np.array(image))
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)
    return image, tensor


def predict(image_source, model=None):
    """
    Run inference on a single image.

    Args:
        image_source: Path to the input image, or a PIL.Image instance
        model: Preloaded model (optional, created if not provided)

    Returns:
        dict with predicted class, label, confidence and full probabilities
    """
    if model is None:
        model = create_model()

    _, tensor = preprocess_image(image_source)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    confidence, predicted_idx = torch.max(probabilities, dim=0)
    predicted_idx = predicted_idx.item()
    predicted_class = IDX_TO_CLASS[predicted_idx]

    return {
        "class": predicted_class,
        "label": CLASS_LABELS[predicted_class],
        "confidence": confidence.item(),
        "probabilities": {
            IDX_TO_CLASS[idx]: probabilities[idx].item()
            for idx in range(NUM_CLASSES)
        }
    }


# ============================================================
# 9. MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]

    result = predict(image_path)

    print("\nPrediction:", result["label"], f"({result['class']})")
    print(f"Confidence: {result['confidence'] * 100:.2f}%")
    print("\nClass probabilities:")
    for class_name, prob in sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True
    ):
        print(f"  {CLASS_LABELS[class_name]:25s}: {prob * 100:.2f}%")
