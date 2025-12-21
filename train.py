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
import yaml
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.core.trainer import Trainer
from src.data.dataset import get_dataset, list_datasets
from src.data.transforms import preprocessing
from src.data.transforms import (
    PreprocessPipeline,
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
        filters=FilterPipeline(filters), norms=NormPipeline(norms), miscs=MiscPipeline(miscs)
    )



def create_data_loaders(dataset_config_path, data_path, train_config, 
                        preprocess_config_path=None, feature_config_path=None):
    """Create train and validation data loaders."""
    # Load dataset config
    dataset_config = load_config(dataset_config_path)
    dataset_cfg = dataset_config.get('dataset', dataset_config)
    
    # Get dataset type and validate
    dataset_type = dataset_cfg.get('type')
    if dataset_type is None:
        raise ValueError(
            f"Dataset config must specify 'type' field (e.g., 'OrionAEFrameDataset' or 'CWTScalogramDataset'). "
            f"Available types: {list_datasets()}. "
            f"Config file: {dataset_config_path}"
        )
    
    LOGGER.info(f"Using dataset type: {dataset_type}")
    
    # Load preprocess config if provided and dataset supports it
    preprocess_pipeline = None
    if preprocess_config_path and preprocess_config_path.exists():
        with open(preprocess_config_path, 'r') as f:
            preprocess_config = yaml.safe_load(f) or {}
        
        preprocess_config_data = preprocess_config.get('preprocess', preprocess_config)
        preprocess_pipeline = create_preprocessing_pipeline(preprocess_config_data)
        LOGGER.info(f"Loaded preprocess config from: {preprocess_config_path}")
    elif preprocess_config_path:
        LOGGER.warning(f"Preprocess config file not found: {preprocess_config_path}. Using empty pipeline.")
        preprocess_pipeline = PreprocessPipeline()
    
    # Build dataset creation kwargs
    train_dataset_kwargs = {
        'data_path': data_path,
        'config_path': str(dataset_config_path),
        'type': 'train',
    }
    
    val_dataset_kwargs = {
        'data_path': data_path,
        'config_path': str(dataset_config_path),
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
    # Only set in_channels if it's already defined in model_params (model accepts it)
    if 'in_channels' in model_params:
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
    
    # Get or generate experiment name
    # Priority: 1) train_config.experiment_name, 2) logging.tensorboard.experiment_name, 3) auto-generate
    experiment_name = train_config.get("experiment_name")
    if experiment_name is None:
        if train_config.get("logging", {}).get("tensorboard", {}).get("experiment_name"):
            experiment_name = train_config["logging"]["tensorboard"]["experiment_name"]
        else:
            from datetime import datetime
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create experiment directory structure: runs/{experiment_name}/
    experiment_dir = Path("runs") / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Experiment directory: {experiment_dir}")
    LOGGER.info(f"All logs and checkpoints will be saved under: {experiment_dir}")
    
    # Update checkpoint save_dir to use experiment directory: runs/{experiment_name}/checkpoints/
    if train_config.get("checkpoint") is not None:
        # Override save_dir to use experiment directory structure
        train_config["checkpoint"]["save_dir"] = str(experiment_dir / "checkpoints")
    
    # Update tensorboard configuration
    # TensorBoardLogger creates: log_dir / experiment_name = runs / {experiment_name}
    if train_config.get("logging", {}).get("tensorboard") is not None:
        train_config["logging"]["tensorboard"]["log_dir"] = "runs"
        train_config["logging"]["tensorboard"]["experiment_name"] = experiment_name
    
    # Update MLflow run_name to match experiment_name if not set
    if train_config.get("logging", {}).get("mlflow") is not None:
        if train_config["logging"]["mlflow"].get("run_name") is None:
            train_config["logging"]["mlflow"]["run_name"] = experiment_name
    
    # Store experiment name in config for reference
    train_config["experiment_name"] = experiment_name
    
    # Save config files to experiment directory for reproducibility
    import shutil
    shutil.copy(train_config_path, experiment_dir / "train_config.yaml")
    shutil.copy(dataset_config_path, experiment_dir / "dataset_config.yaml")
    shutil.copy(model_config_path, experiment_dir / "model_config.yaml")
    if preprocess_config_path and preprocess_config_path.exists():
        shutil.copy(preprocess_config_path, experiment_dir / "preprocess_config.yaml")
    if feature_config_path and feature_config_path.exists():
        shutil.copy(feature_config_path, experiment_dir / "feature_config.yaml")
    LOGGER.info(f"Config files saved to {experiment_dir}")
    
    # Get device
    device_config = train_config.get("device", "cuda")
    device = get_device(device_config)
    train_config["device"] = device
    
    # Create data loaders
    LOGGER.info("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        dataset_config_path=dataset_config_path,
        data_path=str(data_path),
        train_config=train_config,
        preprocess_config_path=preprocess_config_path,
        feature_config_path=feature_config_path,
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

