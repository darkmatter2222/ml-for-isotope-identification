"""
Training Script for Vega Model

Implements the training loop with:
- Mixed precision training for RTX 5090 efficiency
- Learning rate scheduling
- Early stopping
- Model checkpointing
- Training metrics logging
"""

import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .model import VegaModel, VegaConfig, VegaLoss
from .dataset import create_data_loaders, SpectrumDataset
from .isotope_index import IsotopeIndex, get_default_isotope_index


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Data
    data_dir: str = "data/synthetic"
    
    # Model save path
    model_dir: str = "models"
    model_name: str = "vega"
    
    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_epochs: int = 100
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Learning rate scheduling
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    
    # Mixed precision
    use_amp: bool = True
    
    # Data splits
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    
    # Workers
    num_workers: int = 0  # Set to 0 for Windows compatibility
    
    # Reproducibility
    seed: int = 42
    
    # Activity normalization
    max_activity_bq: float = 1000.0


class VegaTrainer:
    """
    Trainer class for the Vega model.
    
    Handles the training loop, validation, checkpointing, and metrics.
    """
    
    def __init__(
        self,
        model: VegaModel,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        force_cpu: bool = False
    ):
        self.model = model
        self.config = config
        
        # Device selection - force CPU if requested or if CUDA incompatible
        if force_cpu:
            self.device = torch.device('cpu')
        elif device:
            self.device = device
        else:
            # Try CUDA but fall back to CPU if there are compatibility issues
            if torch.cuda.is_available():
                try:
                    # Test if CUDA actually works
                    test_tensor = torch.zeros(1, device='cuda')
                    _ = test_tensor + 1
                    self.device = torch.device('cuda')
                except RuntimeError:
                    print("CUDA device found but not compatible, falling back to CPU")
                    self.device = torch.device('cpu')
            else:
                self.device = torch.device('cpu')
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup loss function
        self.loss_fn = VegaLoss(
            classification_weight=model.config.classification_weight,
            regression_weight=model.config.regression_weight
        )
        
        # Setup optimizer
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Setup learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.lr_scheduler_patience,
            factor=config.lr_scheduler_factor,
            min_lr=config.min_lr
        )
        
        # Setup mixed precision training (only if CUDA is working)
        if config.use_amp and self.device.type == 'cuda':
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = []
        
        # Create model directory
        self.model_dir = Path(config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Training on device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Mixed precision: {config.use_amp}")
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        cls_loss_sum = 0.0
        reg_loss_sum = 0.0
        num_batches = 0
        
        for batch in train_loader:
            # Move to device
            spectra = batch['spectrum'].to(self.device)
            presence = batch['presence_labels'].to(self.device)
            activities = batch['activity_labels'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with optional mixed precision
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred_probs, pred_activities = self.model(spectra)
                    loss, loss_dict = self.loss_fn(
                        pred_probs, pred_activities, presence, activities
                    )
                
                # Backward pass with scaling
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                pred_probs, pred_activities = self.model(spectra)
                loss, loss_dict = self.loss_fn(
                    pred_probs, pred_activities, presence, activities
                )
                
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss_dict['total']
            cls_loss_sum += loss_dict['classification']
            reg_loss_sum += loss_dict['regression']
            num_batches += 1
        
        return {
            'train_loss': total_loss / num_batches,
            'train_cls_loss': cls_loss_sum / num_batches,
            'train_reg_loss': reg_loss_sum / num_batches
        }
    
    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate the model."""
        if val_loader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        cls_loss_sum = 0.0
        reg_loss_sum = 0.0
        num_batches = 0
        
        # Metrics
        correct_isotopes = 0
        total_isotopes = 0
        
        for batch in val_loader:
            spectra = batch['spectrum'].to(self.device)
            presence = batch['presence_labels'].to(self.device)
            activities = batch['activity_labels'].to(self.device)
            
            pred_logits, pred_activities = self.model(spectra)
            loss, loss_dict = self.loss_fn(
                pred_logits, pred_activities, presence, activities
            )
            
            total_loss += loss_dict['total']
            cls_loss_sum += loss_dict['classification']
            reg_loss_sum += loss_dict['regression']
            num_batches += 1
            
            # Calculate accuracy (apply sigmoid to logits first)
            pred_probs = torch.sigmoid(pred_logits)
            pred_presence = (pred_probs >= 0.5).float()
            correct_isotopes += (pred_presence == presence).sum().item()
            total_isotopes += presence.numel()
        
        accuracy = correct_isotopes / total_isotopes if total_isotopes > 0 else 0.0
        
        return {
            'val_loss': total_loss / num_batches,
            'val_cls_loss': cls_loss_sum / num_batches,
            'val_reg_loss': reg_loss_sum / num_batches,
            'val_accuracy': accuracy
        }
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save a model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': asdict(self.model.config),
            'training_config': asdict(self.config),
            'training_history': self.training_history
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.parent / f"{self.config.model_name}_best.pt"
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, path: Path):
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(
        self,
        train_loader,
        val_loader,
        resume_from: Optional[Path] = None
    ) -> Dict:
        """
        Full training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            resume_from: Optional path to checkpoint to resume from
            
        Returns:
            Training results dictionary
        """
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        print("\n" + "=" * 60)
        print("Starting Vega Training")
        print("=" * 60)
        print(f"Epochs: {self.config.num_epochs}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        print(f"Training samples: {len(train_loader.dataset)}")
        if val_loader:
            print(f"Validation samples: {len(val_loader.dataset)}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics, 'epoch': epoch}
            self.training_history.append(metrics)
            
            # Update learning rate
            if val_loader and 'val_loss' in val_metrics:
                self.scheduler.step(val_metrics['val_loss'])
            else:
                self.scheduler.step(train_metrics['train_loss'])
            
            # Check for improvement
            val_loss = val_metrics.get('val_loss', train_metrics['train_loss'])
            is_best = val_loss < self.best_val_loss - self.config.min_delta
            
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            checkpoint_path = self.model_dir / f"{self.config.model_name}_epoch_{epoch}.pt"
            self.save_checkpoint(checkpoint_path, is_best=is_best)
            
            # Logging
            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]['lr']
            
            log_str = (
                f"Epoch {epoch+1}/{self.config.num_epochs} | "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
            )
            if val_loader:
                log_str += (
                    f"Val Loss: {val_metrics['val_loss']:.4f} | "
                    f"Val Acc: {val_metrics['val_accuracy']:.4f} | "
                )
            log_str += f"LR: {lr:.2e} | Time: {epoch_time:.1f}s"
            
            if is_best:
                log_str += " *"
            
            print(log_str)
            
            # Early stopping
            if self.epochs_without_improvement >= self.config.patience:
                print(f"\nEarly stopping after {epoch + 1} epochs")
                break
        
        total_time = time.time() - start_time
        
        # Save final model
        final_path = self.model_dir / f"{self.config.model_name}_final.pt"
        self.save_checkpoint(final_path)
        
        # Save training history
        history_path = self.model_dir / f"{self.config.model_name}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        print("\n" + "=" * 60)
        print(f"Training complete!")
        print(f"Total time: {total_time / 60:.1f} minutes")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Model saved to: {final_path}")
        print("=" * 60)
        
        return {
            'best_val_loss': self.best_val_loss,
            'total_epochs': self.current_epoch + 1,
            'total_time': total_time,
            'history': self.training_history
        }


def train_vega(
    data_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    config: Optional[TrainingConfig] = None,
    model_config: Optional[VegaConfig] = None
) -> Tuple[VegaModel, Dict]:
    """
    Convenience function to train a Vega model.
    
    Args:
        data_dir: Path to data directory
        model_dir: Path to save models
        config: Training configuration
        model_config: Model configuration
        
    Returns:
        Tuple of (trained model, training results)
    """
    # Setup paths
    project_root = Path(__file__).parent.parent.parent
    
    if config is None:
        config = TrainingConfig()
    
    if data_dir:
        config.data_dir = data_dir
    if model_dir:
        config.model_dir = model_dir
    
    # Make paths absolute
    data_path = Path(config.data_dir)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    
    model_path = Path(config.model_dir)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    
    config.model_dir = str(model_path)
    
    # Set random seeds
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    
    # Get isotope index
    isotope_index = get_default_isotope_index()
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=data_path,
        batch_size=config.batch_size,
        train_split=config.train_split,
        val_split=config.val_split,
        test_split=config.test_split,
        num_workers=config.num_workers,
        isotope_index=isotope_index,
        max_activity_bq=config.max_activity_bq,
        seed=config.seed
    )
    
    # Create model
    if model_config is None:
        model_config = VegaConfig(
            num_isotopes=isotope_index.num_isotopes,
            max_activity_bq=config.max_activity_bq
        )
    
    model = VegaModel(model_config)
    print(model.summary())
    
    # Create trainer
    trainer = VegaTrainer(model, config)
    
    # Train
    results = trainer.train(train_loader, val_loader)
    
    # Save isotope index with model
    index_path = model_path / f"{config.model_name}_isotope_index.txt"
    isotope_index.save(index_path)
    
    return model, results


if __name__ == "__main__":
    # Quick test training
    model, results = train_vega()
