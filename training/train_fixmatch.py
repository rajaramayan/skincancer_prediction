"""
FixMatch semi-supervised training
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_fixmatch(
    model,
    labeled_loader: DataLoader,
    unlabeled_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    lambda_u: float = 1.0,
    confidence_threshold: float = 0.95,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: Optional[str] = None,
    backbone_name: str = "resnet18",
    num_classes: int = 7,
    resume_from: Optional[str] = None
) -> dict:
    """
    Train model using FixMatch semi-supervised learning
    
    Args:
        model: FixMatch model
        labeled_loader: Labeled data loader
        unlabeled_loader: Unlabeled data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        lambda_u: Weight for unlabeled loss
        confidence_threshold: Confidence threshold for pseudo-labels
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        backbone_name: Name of the backbone architecture (stored for reloading)
        num_classes: Number of output classes (stored for reloading)
        resume_from: Path to a "*_last.pt" checkpoint (model + optimizer + history) to
            resume an interrupted run from, instead of starting at epoch 0
        
    Returns:
        Dictionary with training history
    """
    
    model = model.to(device)
    model.confidence_threshold = confidence_threshold
    
    criterion_labeled = nn.CrossEntropyLoss()
    criterion_unlabeled = nn.MSELoss()  # Consistency regularization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'labeled_loss': [],
        'unlabeled_loss': [],
        'val_loss': [],
        'val_acc': [],
        'pseudo_label_accuracy': []
    }
    
    best_val_acc = -1.0
    start_epoch = 0
    if checkpoint_dir:
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    if resume_from and Path(resume_from).exists():
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        history = ckpt.get('history', history)
        best_val_acc = ckpt.get('best_val_acc', -1.0)
        start_epoch = ckpt.get('epoch', 0)
        print(f"Resumed from {resume_from} (epoch {start_epoch}, best val acc {best_val_acc:.2f}%)\n")
    
    print(f"Training on device: {device}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Epochs: {num_epochs}, Learning rate: {learning_rate}")
    print(f"Lambda U (unlabeled weight): {lambda_u}, Confidence Threshold: {confidence_threshold}\n")
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        labeled_loss = 0.0
        unlabeled_loss = 0.0
        
        # Create iterators for both loaders
        labeled_iter = iter(labeled_loader)
        unlabeled_iter = iter(unlabeled_loader)
        
        # Get max number of iterations
        max_iter = max(len(labeled_loader), len(unlabeled_loader))
        
        train_pbar = tqdm(range(max_iter), desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for _ in train_pbar:
            # Get labeled batch
            try:
                x_labeled, y_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                x_labeled, y_labeled = next(labeled_iter)
            
            x_labeled = x_labeled.to(device)
            y_labeled = y_labeled.to(device)
            
            # Get weakly- and strongly-augmented views of the unlabeled batch
            try:
                x_unlabeled_weak, x_unlabeled_strong = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                x_unlabeled_weak, x_unlabeled_strong = next(unlabeled_iter)
            
            x_unlabeled_weak = x_unlabeled_weak.to(device)
            x_unlabeled_strong = x_unlabeled_strong.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits_labeled, logits_unlabeled_weak, logits_unlabeled_strong = model(
                x_labeled, x_unlabeled_weak, x_unlabeled_strong
            )
            
            # Labeled loss
            loss_labeled = criterion_labeled(logits_labeled, y_labeled)
            
            # Pseudo-label generation and unlabeled loss
            with torch.no_grad():
                probs_weak = torch.softmax(logits_unlabeled_weak, dim=1)
                max_probs, pseudo_labels = torch.max(probs_weak, dim=1)
                mask = max_probs.ge(confidence_threshold).float()
            
            # Consistency regularization loss (MSE between weak and strong predictions)
            probs_strong = torch.softmax(logits_unlabeled_strong, dim=1)
            loss_unlabeled = criterion_unlabeled(probs_strong, probs_weak)
            
            # Total loss
            loss = loss_labeled + lambda_u * loss_unlabeled * mask.mean()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            labeled_loss += loss_labeled.item()
            unlabeled_loss += loss_unlabeled.item()
            
            train_pbar.set_postfix({
                'loss': train_loss / (_ + 1),
                'lab': labeled_loss / (_ + 1),
                'unlab': unlabeled_loss / (_ + 1)
            })
        
        train_loss /= max_iter
        labeled_loss /= max_iter
        unlabeled_loss /= max_iter
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for images, labels in val_pbar:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion_labeled(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': val_loss / (val_total // len(images))})
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['labeled_loss'].append(labeled_loss)
        history['unlabeled_loss'].append(unlabeled_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Labeled Loss: {labeled_loss:.4f}")
        print(f"  Unlabeled Loss: {unlabeled_loss:.4f}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
        
        # Save checkpoint whenever validation accuracy improves
        if checkpoint_dir and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'backbone_name': backbone_name,
                'num_classes': num_classes,
                'val_acc': val_acc,
                'epoch': epoch + 1
            }, Path(checkpoint_dir) / "fixmatch_best.pt")
        
        # Save full resumable state every epoch, so an interrupted run can continue
        # from here instead of restarting at epoch 0
        if checkpoint_dir:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'backbone_name': backbone_name,
                'num_classes': num_classes,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'epoch': epoch + 1,
                'history': history
            }, Path(checkpoint_dir) / "fixmatch_last.pt")
    
    if checkpoint_dir:
        torch.save({
            'model_state_dict': model.state_dict(),
            'backbone_name': backbone_name,
            'num_classes': num_classes,
            'val_acc': history['val_acc'][-1],
            'epoch': num_epochs
        }, Path(checkpoint_dir) / "fixmatch_final.pt")
    
    print("✓ FixMatch Training complete!")
    return history
