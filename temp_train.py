"""
Temporary training script for Orion AE study.
"""
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from src.data.dataset import OrionAEFrameDataset
from src.models import get_model
from src.data.transforms.preprocessing import PreprocessingPipeline, HighPassFilter


def collate_fn(batch):
    """
    Custom collate function to handle dict samples from the dataset.
    Converts numpy arrays to tensors and stacks them.
    """
    # Extract preprocessed data and labels
    preprocessed = [item['preprocessed'] for item in batch]
    labels = [item['label'] for item in batch]
    
    # Convert to tensors
    # preprocessed: list of (time_steps, channels) arrays
    # Stack to (batch_size, time_steps, channels)
    preprocessed_tensor = torch.FloatTensor(np.stack(preprocessed))
    
    # labels: list of ints, convert to tensor
    labels_tensor = torch.LongTensor(labels)
    
    return {
        'preprocessed': preprocessed_tensor,
        'label': labels_tensor
    }


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        # Move data to device
        inputs = batch['preprocessed'].to(device)  # (batch_size, time_steps, channels)
        labels = batch['label'].to(device)  # (batch_size,)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)  # Model should output (batch_size, num_classes)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch in pbar:
            # Move data to device
            inputs = batch['preprocessed'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train model on Orion AE dataset')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--dataset_config', type=str, required=True,
                        help='Path to dataset config YAML file')
    parser.add_argument('--model_config', type=str, required=True,
                        help='Path to model config YAML file')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (cuda/cpu/auto, default: auto)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loader workers (default: 0)')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for TensorBoard logging (default: auto-generated)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Load dataset config
    with open(args.dataset_config, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    # Load model config
    with open(args.model_config, 'r', encoding='utf-8') as f:
        model_config = yaml.safe_load(f)
    
    # Get number of classes from dataset config
    num_classes = len(dataset_config['labels'])
    print(f'Number of classes: {num_classes}')

    prepprocess_pipeline = PreprocessingPipeline(
        filters=[HighPassFilter(cutoff=100, fs=5e6, order=5)]
    )
    
    # Create datasets
    print('Loading datasets...')
    train_dataset = OrionAEFrameDataset(
        data_path=args.data_path,
        config_path=args.dataset_config,
        type='train',
        preprocessing_pipeline=prepprocess_pipeline
    )
    
    val_dataset = OrionAEFrameDataset(
        data_path=args.data_path,
        config_path=args.dataset_config,
        type='val',
        preprocessing_pipeline=prepprocess_pipeline
    )
    
    test_dataset = OrionAEFrameDataset(
        data_path=args.data_path,
        config_path=args.dataset_config,
        type='test',
        preprocessing_pipeline=prepprocess_pipeline
    )
    
    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')
    print(f'Test samples: {len(test_dataset)}')
    
    # Create data loaders
    pin_memory = device.type == 'cuda'
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,  # Change from 0 to 4-8
        pin_memory=pin_memory,  # Add this
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,  # Change from 0
        pin_memory=pin_memory,  # Add this
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,  # Change from 0
        pin_memory=pin_memory,  # Add this
        collate_fn=collate_fn
    )
    
    # Get sample to determine input shape
    sample = train_dataset[0]
    preprocessed_shape = sample['preprocessed'].shape  # (time_steps, channels)
    print(f'Input shape: {preprocessed_shape}')
    
    # Load model
    model_name = model_config['model_name']
    model_params = model_config.get('params', {})
    
    # Add input shape and num_classes to model params if model needs them
    # This is a common pattern, but adjust based on your model implementation
    if 'input_shape' not in model_params:
        model_params['input_shape'] = preprocessed_shape
    if 'num_classes' not in model_params:
        model_params['num_classes'] = num_classes
    
    print(f'Loading model: {model_name} with params: {model_params}')
    model = get_model(model_name, **model_params)
    model = model.to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Setup loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # TensorBoard logging with experiment name
    if args.experiment_name:
        exp_name = args.experiment_name
    else:
        # Auto-generate name: model_name_timestamp
        exp_name = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    log_dir = Path('runs') / exp_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f'TensorBoard logs: {log_dir}')
    print(f'View with: tensorboard --logdir=runs')
    
    # Create checkpoint directory
    checkpoint_dir = log_dir / 'checkpoints'
    checkpoint_dir.mkdir(exist_ok=True)
    
    # Training loop
    print('\nStarting training...')
    best_val_acc = 0.0
    best_epoch = 0
    best_model_state = None
    best_train_loss = None
    best_train_acc = None
    best_val_loss = None
    
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')
        print('-' * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch + 1)
        writer.add_scalar('Loss/Val', val_loss, epoch + 1)
        writer.add_scalar('Accuracy/Train', train_acc, epoch + 1)
        writer.add_scalar('Accuracy/Val', val_acc, epoch + 1)
        
        # Print epoch summary
        print(f'\nEpoch {epoch + 1} Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            
            # Save checkpoint
            checkpoint_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': best_epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': best_val_loss,
                'train_acc': best_train_acc,
                'train_loss': best_train_loss,
            }, checkpoint_path)
            
            print(f'  ✓ New best validation accuracy: {best_val_acc:.2f}% (epoch {best_epoch})')
            print(f'  ✓ Model saved to: {checkpoint_path}')
    
    print('\nTraining completed!')
    print(f'Best validation accuracy: {best_val_acc:.2f}% at epoch {best_epoch}')
    print(f'Best epoch metrics:')
    print(f'  Train Loss: {best_train_loss:.4f}, Train Acc: {best_train_acc:.2f}%')
    print(f'  Val Loss: {best_val_loss:.4f}, Val Acc: {best_val_acc:.2f}%')
    
    # Load best model for test evaluation
    print('\n' + '=' * 50)
    print(f'Loading best model (epoch {best_epoch}) for test evaluation...')
    print('=' * 50)
    model.load_state_dict(best_model_state)
    
    # Evaluate on test set with best model
    print('\nEvaluating on test set...')
    test_loss, test_acc = validate_epoch(model, test_loader, criterion, device)
    
    writer.close()
    
    print(f'\nTest Results (using best model from epoch {best_epoch}):')
    print(f'  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
    print(f'\nBest model checkpoint saved at: {checkpoint_dir / "best_model.pt"}')
    
    # Save training info in human-readable format (JSON)
    training_info = {
        'experiment_name': exp_name,
        'model_name': model_name,
        'model_params': model_params,
        'hyperparameters': {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'num_epochs': args.epochs,
            'num_workers': 4,
            'device': str(device),
        },
        'dataset_info': {
            'data_path': str(args.data_path),
            'num_classes': num_classes,
            'input_shape': list(preprocessed_shape),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'test_samples': len(test_dataset),
        },
        'best_epoch': best_epoch,
        'best_metrics': {
            'train_loss': float(best_train_loss),
            'train_acc': float(best_train_acc),
            'val_loss': float(best_val_loss),
            'val_acc': float(best_val_acc),
        },
        'test_metrics': {
            'test_loss': float(test_loss),
            'test_acc': float(test_acc),
        },
        'model_info': {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
        }
    }
    
    # Save as JSON
    info_file = log_dir / 'training_info.json'
    with open(info_file, 'w', encoding='utf-8') as f:
        json.dump(training_info, f, indent=2, ensure_ascii=False)
    
    print(f'Training info saved to: {info_file}')


if __name__ == '__main__':
    main()

