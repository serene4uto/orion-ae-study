#!/usr/bin/env python3
"""
Training script for Orion AE Study.

Usage:
    python train.py --train_config configs/train/train_example.yaml \
                    --dataset_config configs/dataset/example_1A.yaml \
                    --model_config configs/model/simple_cnn.yaml \
                    --data_path /path/to/data
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.core.trainer import Trainer
from src.data.dataset import OrionAEFrameDataset
from src.data.transforms import preprocessing
from src.data.transforms import (
    PreprocessingPipeline,
    FilterPipeline,
    NormPipeline,
    MiscPipeline,
)
from src.models import get_model
from src.utils import LOGGER


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_config):
    """Get the appropriate device for training."""
    if device_config == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device_config == "cuda":
        if not torch.cuda.is_available():
            LOGGER.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        else:
            device = "cuda"
    else:
        device = device_config
    
    LOGGER.info(f"Using device: {device}")
    return device


def create_preprocessing_pipeline(preprocess_config):
    """
    Create a PreprocessingPipeline from config.
    
    Args:
        preprocess_config: Dictionary with 'filters' and 'norms' keys.
                          Each is a list of transform configs with 'name' and 'params'.
    
    Returns:
        PreprocessingPipeline instance
    """
    if not preprocess_config:
        return PreprocessingPipeline()
    
    # Build filters
    filters = []
    filter_configs = preprocess_config.get('filters', [])
    for filter_cfg in filter_configs:
        if isinstance(filter_cfg, dict):
            transform_type = filter_cfg.get('name')
            params = filter_cfg.get('params', {})
            
            transform_class = getattr(preprocessing, transform_type, None)
            if transform_class is not None:
                filters.append(transform_class(**params))
            else:
                LOGGER.warning(f"Unknown filter type: {transform_type}. Skipping.")
        else:
            LOGGER.warning(f"Invalid filter config format: {filter_cfg}. Expected dict.")
    
    # Build norms
    norms = []
    norm_configs = preprocess_config.get('norms', [])
    for norm_cfg in norm_configs:
        if isinstance(norm_cfg, dict):
            transform_type = norm_cfg.get('name')
            params = norm_cfg.get('params', {})
            
            transform_class = getattr(preprocessing, transform_type, None)
            if transform_class is not None:
                norms.append(transform_class(**params))
            else:
                LOGGER.warning(f"Unknown norm type: {transform_type}. Skipping.")
        else:
            LOGGER.warning(f"Invalid norm config format: {norm_cfg}. Expected dict.")
    
    # Build miscs
    miscs = []
    misc_configs = preprocess_config.get('miscs', [])
    for misc_cfg in misc_configs:
        if isinstance(misc_cfg, dict):
            transform_type = misc_cfg.get('name')
            params = misc_cfg.get('params', {})
            
            transform_class = getattr(preprocessing, transform_type, None)
            if transform_class is not None:
                miscs.append(transform_class(**params))
            else:
                LOGGER.warning(f"Unknown misc type: {transform_type}. Skipping.")
        else:
            LOGGER.warning(f"Invalid misc config format: {misc_cfg}. Expected dict.")
    
    return PreprocessingPipeline(
        filters=FilterPipeline(filters), norms=NormPipeline(norms), miscs=MiscPipeline(miscs)
    )



def create_data_loaders(dataset_config_path, data_path, train_config):
    """Create train and validation data loaders."""
    # Load dataset config to get preprocessing settings
    dataset_config = load_config(dataset_config_path)
    dataset_cfg = dataset_config.get('dataset', dataset_config)
    preprocess_config = dataset_cfg.get('preprocess', {})
    
    # Create preprocessing pipeline from config
    preprocessing_pipeline = create_preprocessing_pipeline(preprocess_config)
    
    # Create train dataset
    train_dataset = OrionAEFrameDataset(
        data_path=data_path,
        config_path=dataset_config_path,
        type='train',
        preprocessing_pipeline=preprocessing_pipeline
    )
    
    # Create validation dataset
    val_dataset = OrionAEFrameDataset(
        data_path=data_path,
        config_path=dataset_config_path,
        type='val',
        preprocessing_pipeline=preprocessing_pipeline
    )
    
    # Get data loader parameters from train config
    batch_size = train_config.get("batch_size", 32)
    num_workers = train_config.get("num_workers", 4)
    pin_memory = train_config.get("pin_memory", True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )
    
    LOGGER.info(f"Train dataset size: {len(train_dataset)}")
    LOGGER.info(f"Validation dataset size: {len(val_dataset)}")
    LOGGER.info(f"Batch size: {batch_size}")
    
    return train_loader, val_loader


def create_model(model_config, dataset_config_path, data_path, device):
    """Create model from configuration."""
    model_config_data = load_config(model_config)
    
    # Extract model config (handle both 'model' root key and direct config)
    if 'model' in model_config_data:
        model_cfg = model_config_data['model']
    else:
        model_cfg = model_config_data
    
    model_name = model_cfg['model_name']
    model_params = model_cfg.get('params', {})
    
    # Get input shape from dataset config
    dataset_config = load_config(dataset_config_path)
    dataset_cfg = dataset_config.get('dataset', dataset_config)
    
    # Get number of channels from dataset config
    channels = dataset_cfg.get('channels', ['A', 'B', 'C', 'D'])
    num_channels = len(channels)
    
    # Get number of classes from dataset config labels
    labels = dataset_cfg.get('labels', {})
    num_classes = len(labels)
    
    # Set num_classes if not specified
    if 'num_classes' not in model_params:
        model_params['num_classes'] = num_classes
    
    # Always override in_channels from dataset config (dataset config is source of truth)
    # This ensures the model matches the actual number of channels in the data
    model_params['in_channels'] = num_channels
    
    LOGGER.info(f"Creating model: {model_name}")
    LOGGER.info(f"Model parameters: {model_params}")
    
    # Create model
    model = get_model(model_name, **model_params)
    model = model.to(device)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"Total parameters: {total_params:,}")
    LOGGER.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Train a model using the Trainer class")
    parser.add_argument(
        "--train_config",
        type=str,
        required=True,
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to dataset configuration YAML file"
    )
    parser.add_argument(
        "--model_config",
        type=str,
        required=True,
        help="Path to model configuration YAML file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the data directory"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    train_config_path = Path(args.train_config)
    dataset_config_path = Path(args.dataset_config)
    model_config_path = Path(args.model_config)
    data_path = Path(args.data_path)
    
    if not train_config_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_config_path}")
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # Load training configuration
    LOGGER.info(f"Loading training config from: {train_config_path}")
    train_config_raw = load_config(train_config_path)
    train_config = train_config_raw.get('train', train_config_raw)
    
    # Get device
    device_config = train_config.get("device", "cuda")
    device = get_device(device_config)
    train_config["device"] = device
    
    # Create data loaders
    LOGGER.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset_config_path=dataset_config_path,
        data_path=str(data_path),
        train_config=train_config
    )
    
    # Create model
    LOGGER.info("Creating model...")
    model = create_model(
        model_config=model_config_path,
        dataset_config_path=dataset_config_path,
        data_path=data_path,
        device=device
    )
    
    # Initialize trainer
    LOGGER.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config
    )
    
    # Start training
    LOGGER.info("Starting training...")
    trainer.train()
    
    LOGGER.info("Training completed successfully!")


if __name__ == "__main__":
    main()

