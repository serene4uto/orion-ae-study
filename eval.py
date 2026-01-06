#!/usr/bin/env python3
"""
Evaluation script for Orion AE Study.

Usage:
    python eval.py --checkpoint runs/experiment_name/checkpoints/best_model.pt \
                   --dataset_config configs/dataset/example_1A.yaml \
                   --model_config configs/model/simple_cnn.yaml \
                   --data_path /path/to/data \
                   [--split test] \
                   [--preprocess_config configs/preprocess/preprocess_example.yaml] \
                   [--output_dir runs/experiment_name/eval_results] \
                   [--device cuda]
    
Note: --split defaults to "test" if not specified, so you typically don't need to include it.
"""

import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)

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
from src.utils import loss
from src.utils import LOGGER

# Try to import seaborn (optional dependency)
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    LOGGER.warning("seaborn not available. Confusion matrix will use matplotlib only.")


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_device(device_config):
    """Get the appropriate device for evaluation."""
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


def load_checkpoint(checkpoint_path, model, device):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model instance to load weights into
        device: Device to load checkpoint on
    
    Returns:
        Dictionary with checkpoint info (epoch, metrics, etc.)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    LOGGER.info(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', -1),
        'metrics': checkpoint.get('metrics', {}),
    }
    
    LOGGER.info(f"Loaded checkpoint from epoch {checkpoint_info['epoch']}")
    if checkpoint_info['metrics']:
        LOGGER.info(f"Checkpoint metrics: {checkpoint_info['metrics']}")
    
    return checkpoint_info


def create_eval_dataset(dataset_config_path, data_path, split, preprocess_config_path=None):
    """
    Create evaluation dataset.
    
    Args:
        dataset_config_path: Path to dataset config YAML
        data_path: Path to data directory
        split: Dataset split ('test', 'val', 'all')
        preprocess_config_path: Optional path to preprocess config
    
    Returns:
        Dataset instance
    """
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
    LOGGER.info(f"Evaluating on split: {split}")
    
    # Load preprocess config if provided
    preprocess_pipeline = None
    if preprocess_config_path and Path(preprocess_config_path).exists():
        with open(preprocess_config_path, 'r') as f:
            preprocess_config = yaml.safe_load(f) or {}
        
        preprocess_config_data = preprocess_config.get('preprocess', preprocess_config)
        preprocess_pipeline = create_preprocessing_pipeline(preprocess_config_data)
        LOGGER.info(f"Loaded preprocess config from: {preprocess_config_path}")
    elif preprocess_config_path:
        LOGGER.warning(f"Preprocess config file not found: {preprocess_config_path}. Using empty pipeline.")
        preprocess_pipeline = PreprocessPipeline()
    
    # Build dataset creation kwargs
    dataset_kwargs = {
        'data_path': data_path,
        'config': dataset_cfg,
        'type': split,
    }
    
    # Add preprocess_pipeline only for OrionAEFrameDataset
    if dataset_type == "OrionAEFrameDataset":
        if preprocess_pipeline is None:
            preprocess_pipeline = PreprocessPipeline()
        dataset_kwargs['preprocess_pipeline'] = preprocess_pipeline
    
    # Create dataset using registry
    dataset = get_dataset(dataset_type, **dataset_kwargs)
    
    LOGGER.info(f"Evaluation dataset size: {len(dataset)}")
    
    return dataset, dataset_cfg


def create_model_from_config(model_config_path, dataset_config_path, device):
    """
    Create model from configuration (same as train.py).
    
    Args:
        model_config_path: Path to model config YAML
        dataset_config_path: Path to dataset config YAML
        device: Device to create model on
    
    Returns:
        Model instance
    """
    model_config_data = load_config(model_config_path)
    
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
    
    # Override in_channels from dataset config if the model accepts it
    if 'in_channels' in model_params:
        model_params['in_channels'] = num_channels
    
    LOGGER.info(f"Creating model: {model_name}")
    LOGGER.info(f"Model parameters: {model_params}")
    
    # Create model
    model = get_model(model_name, **model_params)
    model = model.to(device)
    
    return model, labels


def evaluate_model(model, data_loader, criterion_config, device):
    """
    Evaluate model on dataset.
    
    Args:
        model: Model instance
        data_loader: DataLoader for evaluation
        criterion_config: Loss function configuration (for computing loss)
        device: Device to run evaluation on
    
    Returns:
        Dictionary with predictions, labels, losses, probabilities, and series info
    """
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probabilities = []
    all_series = []
    total_loss = 0.0
    
    # Initialize loss functions (same as training)
    criteria = {}
    for loss_cfg in criterion_config:
        loss_name = loss_cfg.get("name")
        loss_params = loss_cfg.get("params", {})
        
        if loss_name.startswith("nn."):
            loss_name = loss_name[3:]
            if not hasattr(torch.nn, loss_name):
                raise AttributeError(f"torch.nn has no attribute '{loss_name}'")
            loss_class = getattr(torch.nn, loss_name)(**loss_params)
        else:
            if not hasattr(loss, loss_name):
                raise AttributeError(f"loss module has no attribute '{loss_name}'")
            loss_class = getattr(loss, loss_name)(**loss_params)
        
        criteria[loss_name] = loss_class
    
    # Use first loss function for evaluation loss
    criterion = list(criteria.values())[0] if criteria else torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        pbar = tqdm(data_loader, desc='Evaluating')
        for batch in pbar:
            inputs = batch['final'].to(device, dtype=torch.float32)
            labels = batch['label'].to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss_value = criterion(outputs, labels)
            total_loss += loss_value.item()
            
            # Get predictions
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            # Store results
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            
            # Store series info if available
            if 'serie' in batch:
                series_batch = batch['serie']
                if isinstance(series_batch, list):
                    all_series.extend(series_batch)
                else:
                    all_series.extend(series_batch.cpu().numpy().tolist())
    
    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0.0
    
    results = {
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probabilities),
        'series': all_series if all_series else None,
        'loss': avg_loss,
    }
    
    return results


def compute_metrics(predictions, labels, label_names):
    """
    Compute classification metrics.
    
    Args:
        predictions: Array of predicted labels
        labels: Array of true labels
        label_names: List of label names (for class names)
    
    Returns:
        Dictionary with computed metrics
    """
    accuracy = accuracy_score(labels, predictions)
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, zero_division=0
    )
    
    # Classification report
    report = classification_report(
        labels, predictions,
        target_names=label_names,
        output_dict=True,
        zero_division=0
    )
    
    # Per-class accuracy (diagonal of normalized confusion matrix)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    per_class_accuracy = cm_normalized.diagonal()
    
    metrics = {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': per_class_accuracy.tolist(),
        'per_class_precision': precision.tolist(),
        'per_class_recall': recall.tolist(),
        'per_class_f1': f1.tolist(),
        'per_class_support': support.tolist(),
        'classification_report': report,
    }
    
    return metrics


def plot_confusion_matrix(cm, label_names, output_path):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix array
        label_names: List of label names
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    if HAS_SEABORN:
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=label_names,
            yticklabels=label_names,
            cbar_kws={'label': 'Normalized Count'}
        )
    else:
        # Fallback to matplotlib imshow
        plt.imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        plt.colorbar(label='Normalized Count')
        tick_marks = np.arange(len(label_names))
        plt.xticks(tick_marks, label_names, rotation=45)
        plt.yticks(tick_marks, label_names)
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                plt.text(j, i, format(cm_normalized[i, j], '.2f'),
                        horizontalalignment="center",
                        color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.title('Confusion Matrix (Normalized)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    LOGGER.info(f"Saved confusion matrix to: {output_path}")


def save_results(results, metrics, label_names, output_dir, checkpoint_info=None):
    """
    Save evaluation results to files.
    
    Args:
        results: Dictionary with predictions, labels, probabilities, etc.
        metrics: Dictionary with computed metrics
        label_names: List of label names
        output_dir: Directory to save results
        checkpoint_info: Optional checkpoint information
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics_path = output_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'checkpoint_info': checkpoint_info,
            'metrics': metrics,
            'label_names': label_names,
        }, f, indent=2)
    LOGGER.info(f"Saved metrics to: {metrics_path}")
    
    # Save classification report as text
    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write("Classification Report\n")
        f.write("=" * 50 + "\n\n")
        if checkpoint_info:
            f.write(f"Checkpoint Epoch: {checkpoint_info.get('epoch', 'N/A')}\n")
            f.write(f"Checkpoint Metrics: {checkpoint_info.get('metrics', {})}\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n\n")
        f.write("Per-Class Metrics:\n")
        f.write("-" * 50 + "\n")
        for i, label_name in enumerate(label_names):
            f.write(f"{label_name}:\n")
            f.write(f"  Accuracy: {metrics['per_class_accuracy'][i]:.4f}\n")
            f.write(f"  Precision: {metrics['per_class_precision'][i]:.4f}\n")
            f.write(f"  Recall: {metrics['per_class_recall'][i]:.4f}\n")
            f.write(f"  F1-Score: {metrics['per_class_f1'][i]:.4f}\n")
            f.write(f"  Support: {metrics['per_class_support'][i]}\n\n")
        f.write("\nDetailed Classification Report:\n")
        f.write("-" * 50 + "\n")
        f.write(classification_report(
            results['labels'],
            results['predictions'],
            target_names=label_names,
            zero_division=0
        ))
    LOGGER.info(f"Saved classification report to: {report_path}")
    
    # Save predictions as CSV
    predictions_df = pd.DataFrame({
        'true_label': results['labels'],
        'predicted_label': results['predictions'],
        'true_label_name': [label_names[l] for l in results['labels']],
        'predicted_label_name': [label_names[p] for p in results['predictions']],
    })
    
    # Add series if available
    if results['series']:
        predictions_df['series'] = results['series']
    
    # Add probabilities for each class
    for i, label_name in enumerate(label_names):
        predictions_df[f'prob_{label_name}'] = results['probabilities'][:, i]
    
    # Add max probability
    predictions_df['max_probability'] = results['probabilities'].max(axis=1)
    
    # Add correct flag
    predictions_df['correct'] = (predictions_df['true_label'] == predictions_df['predicted_label'])
    
    predictions_path = output_dir / 'predictions.csv'
    predictions_df.to_csv(predictions_path, index=False)
    LOGGER.info(f"Saved predictions to: {predictions_path}")
    
    # Plot confusion matrix
    cm = np.array(metrics['confusion_matrix'])
    cm_path = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(cm, label_names, cm_path)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint file (e.g., runs/experiment/checkpoints/best_model.pt)"
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
        "--split",
        type=str,
        default="test",
        choices=["test", "val", "all"],
        help="Dataset split to evaluate on. Defaults to 'test' if not specified. Options: test, val, all"
    )
    parser.add_argument(
        "--preprocess_config",
        type=str,
        default=None,
        help="Path to preprocess configuration YAML file (optional, only for OrionAEFrameDataset)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: checkpoint directory / eval_results)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use for evaluation (default: auto)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation (default: 32)"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading (default: 4)"
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default=None,
        help="Path to training config YAML (optional, for loss function config)"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    checkpoint_path = Path(args.checkpoint)
    dataset_config_path = Path(args.dataset_config)
    model_config_path = Path(args.model_config)
    data_path = Path(args.data_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Data path not found: {data_path}")
    
    # Get device
    device = get_device(args.device)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: checkpoint directory / eval_results
        output_dir = checkpoint_path.parent.parent / 'eval_results' / args.split
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Output directory: {output_dir}")
    
    # Load training config for loss function (if provided)
    criterion_config = None
    if args.train_config:
        train_config_path = Path(args.train_config)
        if train_config_path.exists():
            train_config = load_config(train_config_path)
            train_cfg = train_config.get('train', train_config)
            criterion_config = train_cfg.get('criterion', [{'name': 'nn.CrossEntropyLoss', 'weight': 1.0, 'params': {}}])
            LOGGER.info(f"Loaded loss config from: {train_config_path}")
        else:
            LOGGER.warning(f"Train config not found: {train_config_path}. Using default CrossEntropyLoss.")
    
    # Use default loss if not provided
    if criterion_config is None:
        criterion_config = [{'name': 'nn.CrossEntropyLoss', 'weight': 1.0, 'params': {}}]
    
    # Create model
    LOGGER.info("Creating model...")
    model, labels = create_model_from_config(model_config_path, dataset_config_path, device)
    
    # Load checkpoint
    checkpoint_info = load_checkpoint(checkpoint_path, model, device)
    
    # Create evaluation dataset
    LOGGER.info("Creating evaluation dataset...")
    dataset, dataset_cfg = create_eval_dataset(
        dataset_config_path=dataset_config_path,
        data_path=str(data_path),
        split=args.split,
        preprocess_config_path=args.preprocess_config,
    )
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Evaluate model
    LOGGER.info("Running evaluation...")
    results = evaluate_model(model, data_loader, criterion_config, device)
    
    # Get label names
    label_names = list(labels.keys())
    
    # Compute metrics
    LOGGER.info("Computing metrics...")
    metrics = compute_metrics(results['predictions'], results['labels'], label_names)
    
    # Print summary
    LOGGER.info("=" * 50)
    LOGGER.info("Evaluation Results")
    LOGGER.info("=" * 50)
    LOGGER.info(f"Dataset split: {args.split}")
    LOGGER.info(f"Dataset size: {len(dataset)}")
    LOGGER.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    LOGGER.info(f"Average Loss: {results['loss']:.4f}")
    LOGGER.info("\nPer-Class Accuracy:")
    for i, label_name in enumerate(label_names):
        LOGGER.info(f"  {label_name}: {metrics['per_class_accuracy'][i]:.4f}")
    LOGGER.info("=" * 50)
    
    # Save results
    LOGGER.info("Saving results...")
    save_results(results, metrics, label_names, output_dir, checkpoint_info)
    
    # Save evaluation config for reproducibility
    eval_config = {
        'checkpoint': str(checkpoint_path),
        'dataset_config': str(dataset_config_path),
        'model_config': str(model_config_path),
        'data_path': str(data_path),
        'split': args.split,
        'preprocess_config': str(args.preprocess_config) if args.preprocess_config else None,
        'device': device,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'timestamp': datetime.now().isoformat(),
    }
    config_path = output_dir / 'eval_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(eval_config, f, default_flow_style=False)
    LOGGER.info(f"Saved evaluation config to: {config_path}")
    
    LOGGER.info("Evaluation completed successfully!")


if __name__ == "__main__":
    main()

