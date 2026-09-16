# Skin Cancer Classification

ResNet18-based classifier trained on the HAM10000 skin lesion dataset, supporting both
supervised and FixMatch (semi-supervised) training.

## Project Structure

```
inference.py                 # Run predictions on a single image
app.py                        # Streamlit web UI (upload image, view prediction)
explainability.py             # Grad-CAM and SHAP explanation utilities
models/
  backbone.py                 # create_backbone(): resnet18/34/50, densenet, mobilenet
  supervised.py                # SupervisedModel wrapper
  fixmatch.py                  # FixMatch model/training utilities
  checkpoints/                 # Saved model weights (supervised_best/last/final.pt)
preprocessing/
  dataset.py                   # Dataset class + CLASS_TO_IDX label mapping
  split_data.py                 # Labeled/unlabeled/validation split for HAM10000 metadata
  transforms.py                 # Train/test augmentation pipelines
training/
  train_supervised.py            # train_supervised(): supervised training loop
  train_fixmatch.py               # FixMatch semi-supervised training loop
  pseudo_labels.py                 # Pseudo-label generation for unlabeled data
```

## Setup

1. Create/activate a virtual environment (already present as `.venv`).
2. Install dependencies:
   ```powershell
   pip install torch torchvision numpy pillow pandas scikit-learn tqdm albumentations streamlit matplotlib shap
   ```

## Running Inference

```powershell
python inference.py <path_to_image>
```

This loads `models/checkpoints/supervised_best.pt`, runs the image through the resnet18
backbone, and prints the predicted class, confidence, and per-class probabilities.

## Running the Web App (Streamlit)

```powershell
streamlit run app.py
```

This starts a local web server (default: [http://localhost:8501](http://localhost:8501))
where you can upload an image via the browser and see the predicted class, confidence,
and a probability chart. Stop it with `Ctrl+C` in the terminal.

### Explainability (Grad-CAM / SHAP)

After a prediction is shown, expand the **Grad-CAM** or **SHAP** tab and click the
generate button to view a heatmap overlay of the image regions that most influenced
the predicted class:

- **Grad-CAM**: gradient-weighted class activation map from the last conv block
  (`model.backbone.layer4[-1]`). Fast, deterministic.
- **SHAP**: `shap.GradientExplainer` using noisy variants of the input image as a
  background (no training set is available at inference time, so this is a
  lightweight approximation). Slower than Grad-CAM.

Both are also available programmatically via `explainability.generate_gradcam()`
and `explainability.generate_shap_explanation()`.

Class labels (from `preprocessing/dataset.py`):

| Code  | Label |
|-------|-------|
| nv    | Nevus |
| mel   | Melanoma |
| bkl   | Benign Keratosis |
| bcc   | Basal Cell Carcinoma |
| akiec | Actinic Keratosis |
| vasc  | Vascular Lesion |
| df    | Dermatofibroma |

## Training

The `training/` modules are library functions (no CLI entry point) meant to be called
from your own script or notebook, e.g.:

```python
from models.backbone import create_backbone
from models.supervised import SupervisedModel
from training.train_supervised import train_supervised

backbone = create_backbone("resnet18", pretrained=True, num_classes=7)
model = SupervisedModel(backbone, num_classes=7)

history = train_supervised(
    model, train_loader, val_loader,
    num_epochs=10, checkpoint_dir="models/checkpoints"
)
```

Use `preprocessing/split_data.py` to generate labeled/unlabeled/validation splits from
the HAM10000 metadata CSV before building your data loaders.
