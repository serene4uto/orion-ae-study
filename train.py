#!/usr/bin/env python3
"""
Training script for Orion AE Study.

Usage:
    python train.py --train_config configs/train/train_example.yaml \
                    --dataset_config configs/dataset/example_1A.yaml \
                    --model_config configs/model/simple_cnn.yaml \
                    --data_path /path/to/data \
                    [--preprocess_config configs/preprocess/preprocess_example.yaml] \
                    [--feature_config configs/feature/feature_example.yaml]
"""

import argparse
import inspect
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.core.trainer import Trainer
from src.data.dataset import get_dataset, list_datasets
from src.data.transforms import preprocessing
from src.data.transforms import PreprocessPipeline
from src.models import get_model, MODEL_REGISTRY
from src.utils import LOGGER


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, config_path):
    """Save YAML configuration file."""
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

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
    Create a PreprocessPipeline from config.
    
    Args:
        preprocess_config: Dictionary with 'filters' and 'norms' keys.
                          Each is a list of transform configs with 'name' and 'params'.
    
    Returns:
        PreprocessPipeline instance
    """
    if not preprocess_config:
        return PreprocessPipeline()
    
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
    
    return PreprocessPipeline(
        filters=filters, norms=norms, miscs=miscs
    )

def create_data_loaders(dataset_config, data_path, train_config, 
                        preprocess_config=None, feature_config=None):
    """Create train and validation data loaders."""
    
    dataset_cfg = dataset_config.get('dataset', dataset_config)
    
    # Get dataset type and validate
    dataset_type = dataset_cfg.get('type')
    if dataset_type is None:
        raise ValueError(
            f"Dataset config must specify 'type' field (e.g., 'OrionAEFrameDataset' or 'CWTScalogramDataset'). "
            f"Available types: {list_datasets()}. "
        )
    
    LOGGER.info(f"Using dataset type: {dataset_type}")
    
    # Create preprocessing pipeline
    preprocess_pipeline = create_preprocessing_pipeline(preprocess_config)
    
    # Build dataset creation kwargs
    train_dataset_kwargs = {
        'data_path': data_path,
        'config': dataset_config,
        'type': 'train',
    }
    
    val_dataset_kwargs = {
        'data_path': data_path,
        'config': dataset_config,
        'type': 'val',
    }
    
    # Add preprocess_pipeline only for OrionAEFrameDataset
    if dataset_type == "OrionAEFrameDataset":
        if preprocess_pipeline is None:
            preprocess_pipeline = PreprocessPipeline()
        train_dataset_kwargs['preprocess_pipeline'] = preprocess_pipeline
        val_dataset_kwargs['preprocess_pipeline'] = preprocess_pipeline
    
    # Create datasets using registry
    train_dataset = get_dataset(dataset_type, **train_dataset_kwargs)
    val_dataset = get_dataset(dataset_type, **val_dataset_kwargs)
    
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


def create_model(model_config, dataset_config, data_path, device, train_dataset=None):
    """Create model from configuration."""
    model_config_data = model_config.get('model', model_config)
    model_name = model_config_data['model_name']
    model_params = model_config_data.get('params', {}).copy()  # Copy to avoid modifying original
    
    # Get number of classes from dataset config labels
    labels = dataset_config.get('labels', {})
    num_classes = len(labels)
    
    # Set num_classes if not specified
    if 'num_classes' not in model_params:
        model_params['num_classes'] = num_classes
    
    # Auto-detect in_channels from actual data if model accepts it
    model_class = MODEL_REGISTRY.get(model_name)
    if model_class is not None:
        sig = inspect.signature(model_class.__init__)
        if 'in_channels' in sig.parameters:
            if train_dataset is not None and len(train_dataset) > 0:
                # Auto-detect from actual data shape
                sample = train_dataset[0]
                in_channels = sample['final'].shape[0]  # Shape: (C, H, W) or (C, 1, T)
                model_params['in_channels'] = in_channels
                LOGGER.info(f"Auto-detected in_channels={in_channels} from dataset sample")
            else:
                # Fallback to dataset config channels
                channels = dataset_config.get('channels', ['A', 'B', 'C'])
                model_params['in_channels'] = len(channels)
                LOGGER.info(f"Set in_channels={len(channels)} from dataset config")
    
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
    parser.add_argument(
        "--preprocess_config",
        type=str,
        default=None,
        help="Path to preprocess configuration YAML file (optional, only for OrionAEFrameDataset)"
    )
    parser.add_argument(
        "--feature_config",
        type=str,
        default=None,
        help="Path to feature configuration YAML file (optional, currently unused in training)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    train_config_path = Path(args.train_config)
    dataset_config_path = Path(args.dataset_config)
    model_config_path = Path(args.model_config)
    data_path = Path(args.data_path)
    preprocess_config_path = Path(args.preprocess_config) if args.preprocess_config else None
    feature_config_path = Path(args.feature_config) if args.feature_config else None
    
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
    
    # Load dataset configuration
    LOGGER.info(f"Loading dataset config from: {dataset_config_path}")
    dataset_config_raw = load_config(dataset_config_path)
    dataset_config = dataset_config_raw.get('dataset', dataset_config_raw)
    
    # Load model configuration
    LOGGER.info(f"Loading model config from: {model_config_path}")
    model_config_raw = load_config(model_config_path)
    model_config = model_config_raw.get('model', model_config_raw)
    
    # Load preprocess configuration
    preprocess_config = None
    if preprocess_config_path:
        LOGGER.info(f"Loading preprocess config from: {preprocess_config_path}")
        preprocess_config = load_config(preprocess_config_path).get('preprocess', {})
    
    # Load feature configuration
    feature_config = None
    if feature_config_path:
        LOGGER.info(f"Loading feature config from: {feature_config_path}")
        feature_config = load_config(feature_config_path).get('feature', {})
    
    # Get or generate experiment name
    # Priority: 1) train_config.experiment_name, 2) auto-generate
    if train_config.get("experiment_name") is None:
        from datetime import datetime
        train_config["experiment_name"] = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create experiment directory structure: runs/{experiment_name}/
    runs_dir = train_config.get("run_dir") or "runs"
    experiment_dir = Path(runs_dir) / train_config["experiment_name"]

    # Get device
    device_config = train_config.get("device", "cuda")
    device = get_device(device_config)
    train_config["device"] = device
    
    # Create data loaders
    LOGGER.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset_config=dataset_config,
        data_path=str(data_path),
        train_config=train_config,
        preprocess_config=preprocess_config,
        feature_config=feature_config,
    )
    
    # Create model (pass dataset for auto-detecting in_channels)
    LOGGER.info("Creating model...")
    model = create_model(
        model_config=model_config,
        dataset_config=dataset_config,
        data_path=data_path,
        device=device,
        train_dataset=train_loader.dataset
    )
    
    # Initialize trainer
    LOGGER.info("Initializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        experiment_dir=experiment_dir
    )
    
    # Start training
    LOGGER.info("Starting training...")
    trainer.train()
    
    LOGGER.info("Training completed successfully!")
    
    # Saving the configuration for reproducibility
    save_config_path = experiment_dir / "config"
    save_config(train_config, Path(save_config_path) / "train_config.yaml")
    save_config(dataset_config, Path(save_config_path) / "dataset_config.yaml")
    save_config(model_config, Path(save_config_path) / "model_config.yaml")
    if preprocess_config_path and preprocess_config_path.exists():
        save_config(preprocess_config, Path(save_config_path) / "preprocess_config.yaml")
    if feature_config_path and feature_config_path.exists():
        save_config(feature_config, Path(save_config_path) / "feature_config.yaml")
    LOGGER.info(f"Config files saved to {save_config_path}")
    

if __name__ == "__main__":
    main()

