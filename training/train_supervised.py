"""
Supervised baseline training
"""

from pathlib import Path
from typing import Tuple, Optional, Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_supervised(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    checkpoint_dir: Optional[str] = None,
    backbone_name: str = "resnet18",
    num_classes: int = 7,
    resume_from: Optional[str] = None
) -> dict:
    """
    Train supervised baseline model
    
    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
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
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
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
    print(f"Epochs: {num_epochs}, Learning rate: {learning_rate}\n")
    
    for epoch in range(start_epoch, num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_pbar:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': train_loss / (train_total // len(images))})
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
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
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': val_loss / (val_total // len(images))})
        
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
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
            }, Path(checkpoint_dir) / "supervised_best.pt")

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
            }, Path(checkpoint_dir) / "supervised_last.pt")
    
    if checkpoint_dir:
        torch.save({
            'model_state_dict': model.state_dict(),
            'backbone_name': backbone_name,
            'num_classes': num_classes,
            'val_acc': history['val_acc'][-1],
            'epoch': num_epochs
        }, Path(checkpoint_dir) / "supervised_final.pt")
    
    print("✓ Training complete!")
    return history
