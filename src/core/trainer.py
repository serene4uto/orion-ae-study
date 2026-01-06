import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path
from datetime import datetime

from src.utils import loss
from src.utils import LOGGER

class CheckpointManager:
    """Manages model checkpointing during training."""
    def __init__(self, save_dir, monitor="val/loss", save_best=True, save_every_n_epochs=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.save_best = save_best
        self.save_every_n_epochs = save_every_n_epochs
        
        # Determine if metric should be minimized (loss) or maximized (acc)
        self.mode = "min" if "loss" in monitor.lower() else "max"
        self.best_metric = float('inf') if self.mode == "min" else float('-inf')
        self.best_epoch = -1
    
    def should_save_best(self, current_metric):
        """Check if current metric is better than best."""
        if self.mode == "min":
            return current_metric < self.best_metric
        else:
            return current_metric > self.best_metric
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch, metrics, is_best=False):
        """Save a checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        LOGGER.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save best model if applicable
        if is_best and self.save_best:
            best_path = self.save_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            LOGGER.info(f"Saved best model to {best_path}")
    
    def __call__(self, model, optimizer, scheduler, epoch, metrics):
        """Check if checkpoint should be saved and save if needed."""
        current_metric = metrics.get(self.monitor)
        if current_metric is None:
            LOGGER.warning(f"Monitor metric '{self.monitor}' not found in metrics. Skipping checkpoint.")
            return
        
        is_best = False
        if self.should_save_best(current_metric):
            self.best_metric = current_metric
            self.best_epoch = epoch
            is_best = True
        
        # Check if this is a periodic checkpoint epoch
        is_periodic = (self.save_every_n_epochs is not None and 
                      (epoch + 1) % self.save_every_n_epochs == 0)
        
        # Save best model if improved
        if is_best and self.save_best:
            self.save_checkpoint(model, optimizer, scheduler, epoch, metrics, is_best=True)
        # Save periodic checkpoints (only if not already saved as best)
        elif is_periodic:
            self.save_checkpoint(model, optimizer, scheduler, epoch, metrics, is_best=False)


class TensorBoardLogger:
    """TensorBoard logging wrapper."""
    def __init__(self, config, log_dir):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("tensorboard is required. Install with: pip install tensorboard")
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(str(self.log_dir))
        self.log_interval = config.get("log_interval", 10)
        self.log_graph = config.get("log_graph", False)
        self.log_histograms = config.get("log_histograms", False)
        self.batch_count = 0
        
        LOGGER.info(f"TensorBoard logging initialized at {self.log_dir}")
    
    def log_metrics(self, metrics, epoch, step=None):
        """Log metrics to TensorBoard."""
        if step is None:
            step = epoch
        
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
    
    def log_model_graph(self, model, input_shape):
        """Log model computational graph."""
        if self.log_graph:
            try:
                dummy_input = torch.randn(1, *input_shape).to(next(model.parameters()).device)
                self.writer.add_graph(model, dummy_input)
            except Exception as e:
                LOGGER.warning(f"Failed to log model graph: {e}")
    
    def log_histograms(self, model, epoch):
        """Log weight histograms."""
        if self.log_histograms:
            for name, param in model.named_parameters():
                self.writer.add_histogram(name, param, epoch)
    
    def close(self):
        """Close the writer."""
        self.writer.close()


class MLflowLogger:
    """MLflow logging wrapper."""
    def __init__(self, config):
        try:
            import mlflow
            import mlflow.pytorch
        except ImportError:
            raise ImportError("mlflow is required. Install with: pip install mlflow")
        
        self.config = config
        tracking_uri = config.get("tracking_uri")
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        
        experiment_name = config.get("experiment_name", "default")
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception:
            experiment_id = mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        
        run_name = config.get("run_name")
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run = mlflow.start_run(run_name=run_name)
        self.log_artifacts = config.get("log_artifacts", True)
        self.should_log_model = config.get("log_model", True)  # Renamed to avoid conflict
        self.tags = config.get("tags", {})
        self.metrics_prefix = config.get("metrics_prefix", "")
        
        # Set tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)
        
        LOGGER.info(f"MLflow logging initialized. Run: {self.run.info.run_name}")
    
    def log_params(self, params):
        """Log hyperparameters."""
        import mlflow
        for key, value in params.items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)
            else:
                mlflow.log_param(key, str(value))
    
    def log_metrics(self, metrics, epoch):
        """Log metrics."""
        import mlflow
        for key, value in metrics.items():
            metric_name = f"{self.metrics_prefix}{key}" if self.metrics_prefix else key
            mlflow.log_metric(metric_name, value, step=epoch)
    
    def log_artifacts_dir(self, dir_path, artifact_path=None):
        """Log directory as artifacts."""
        if self.log_artifacts:
            import mlflow
            mlflow.log_artifacts(dir_path, artifact_path)
    
    def log_model(self, model, artifact_path="model"):
        """Log PyTorch model."""
        if self.should_log_model:  # Use renamed variable
            import mlflow.pytorch
            mlflow.pytorch.log_model(model, artifact_path)
    
    def close(self):
        """End MLflow run."""
        import mlflow
        mlflow.end_run()


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, monitor="val/loss", mode="auto"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        # Auto-detect mode if not specified
        if mode == "auto":
            self.mode = "min" if "loss" in monitor.lower() else "max"
        
        self.best_metric = float('inf') if self.mode == "min" else float('-inf')
        self.patience_counter = 0
        self.best_epoch = 0
        self.stopped = False
    
    def __call__(self, current_metric, epoch):
        if self.mode == "min":
            improved = current_metric < (self.best_metric - self.min_delta)
        else:  # mode == "max"
            improved = current_metric > (self.best_metric + self.min_delta)
        
        if improved:
            self.best_metric = current_metric
            self.best_epoch = epoch
            self.patience_counter = 0
            return False  # Don't stop
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.stopped = True
                return True  # Stop training
            return False  # Don't stop yet

class Trainer:
    
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        **kwargs,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.config = config
        self.device = config["device"]
        self.optimizer_cfg = config["optimizer"]
        self.criterion_cfg = config["criterion"]
        self.scheduler_cfg = config["scheduler"]
        
        self.epochs = config.get("epochs") or config.get("num_epochs", 100)
        self.criteria_weights = [c['weight'] for c in config["criterion"]]
        self.l1_reg = config.get("l1_reg", 0.0)
        self.early_stopping = config.get("early_stopping")
        self.max_grad_norm = config.get("max_grad_norm", None)
        self.checkpoint_cfg = config.get("checkpoint")
        self.logging_cfg = config.get("logging")
        
        self.experiment_dir = Path(kwargs.get("experiment_dir"))
        
        
        # Initialize early stopping
        if self.early_stopping is not None:
            self.early_stopper = EarlyStopping(
                patience=self.early_stopping.get("patience", 10),
                min_delta=self.early_stopping.get("min_delta", 0.0),
                monitor=self.early_stopping.get("monitor", "val/loss")
            )
        else:
            self.early_stopper = None

        # Initialize checkpointing
        self.checkpoint_manager = None
        if self.checkpoint_cfg is not None:
            self.checkpoint_manager = CheckpointManager(
                save_dir=self.experiment_dir / "checkpoints",
                monitor=self.checkpoint_cfg.get("monitor", "val/loss"),
                save_best=self.checkpoint_cfg.get("save_best", True),
                save_every_n_epochs=self.checkpoint_cfg.get("save_every_n_epochs", None)
            )

        # Initialize loggers
        log_dir = self.experiment_dir / "logs"
        self.loggers = []
        if self.logging_cfg is not None:
            if self.logging_cfg.get("tensorboard", {}).get("enabled", False):
                self.loggers.append(TensorBoardLogger(self.logging_cfg["tensorboard"], log_dir))
            if self.logging_cfg.get("mlflow", {}).get("enabled", False):
                self.loggers.append(MLflowLogger(self.logging_cfg["mlflow"]))

        self.criteria = self._init_criteria()
        self.optimizer = self._init_optimizer()
        self.scheduler = self._init_scheduler()

    def _init_scheduler(self):
        # Scheduler is optional, return None if not configured
        if self.scheduler_cfg is None:
            return None

        scheduler_name = self.scheduler_cfg.get("name")
        if not scheduler_name:
            raise ValueError("Scheduler configuration must include 'name' field")

        # Schedulers are in torch.optim.lr_scheduler
        if not hasattr(optim.lr_scheduler, scheduler_name):
            raise AttributeError(f"torch.optim.lr_scheduler has no attribute '{scheduler_name}'")

        scheduler_class = getattr(optim.lr_scheduler, scheduler_name)

        # Get scheduler parameters
        scheduler_params = self.scheduler_cfg.get("params", {}).copy()  # Copy to avoid modifying original
        if not isinstance(scheduler_params, dict):
            raise ValueError("Scheduler 'params' must be a dictionary")

        # Special handling for OneCycleLR: calculate total_steps if not provided
        if scheduler_name == "OneCycleLR":
            if "total_steps" not in scheduler_params or scheduler_params["total_steps"] is None:
                # Calculate total_steps: epochs * batches_per_epoch
                batches_per_epoch = len(self.train_loader)
                total_steps = self.epochs * batches_per_epoch
                scheduler_params["total_steps"] = total_steps
                LOGGER.info(
                    f"OneCycleLR: calculated total_steps = {total_steps} "
                    f"({self.epochs} epochs × {batches_per_epoch} batches/epoch)"
                )

        # Create scheduler with optimizer as first argument
        # Note: Some schedulers like ReduceLROnPlateau have different signatures,
        # but most follow this pattern
        scheduler_instance = scheduler_class(
            self.optimizer,
            **scheduler_params
        )

        return scheduler_instance


    def _init_optimizer(self):
        optimizer_name = self.optimizer_cfg.get("name")
        if not optimizer_name:
            raise ValueError("Optimizer configuration must include 'name' field")

        if not hasattr(optim, optimizer_name):
            raise AttributeError(f"torch.optim has no attribute '{optimizer_name}'")

        optimizer_class = getattr(optim, optimizer_name)

        # Get learning rate (required parameter)
        lr = self.optimizer_cfg.get("lr")
        if lr is None:
            raise ValueError("Optimizer configuration must include 'lr' field")
        # Ensure lr is numeric
        lr = float(lr)

        # Get additional parameters
        optimizer_params = self.optimizer_cfg.get("params", {})
        if optimizer_params is None:
            optimizer_params = {}
        if not isinstance(optimizer_params, dict):
            raise ValueError("Optimizer 'params' must be a dictionary")
        
        # Convert string numbers to proper types
        converted_params = {}
        for key, value in optimizer_params.items():
            if isinstance(value, str):
                # Try to convert string to number
                try:
                    # Try float first (handles both int and float strings)
                    if '.' in value or 'e' in value.lower() or 'E' in value:
                        converted_params[key] = float(value)
                    else:
                        # Try int, fallback to float
                        try:
                            converted_params[key] = int(value)
                        except ValueError:
                            converted_params[key] = float(value)
                except ValueError:
                    # If conversion fails, keep as string (might be a valid string param)
                    converted_params[key] = value
            elif isinstance(value, list):
                # Convert list elements if they're strings
                converted_params[key] = [
                    float(v) if isinstance(v, str) and ('.' in v or 'e' in v.lower() or 'E' in v) 
                    else (int(v) if isinstance(v, str) else v)
                    for v in value
                ]
            else:
                converted_params[key] = value

        # Create optimizer with model parameters
        optimizer_instance = optimizer_class(
            self.model.parameters(),
            lr=lr,
            **converted_params
        )

        return optimizer_instance

    def _init_criteria(self):
        criteria = {}
        for loss_cfg in self.criterion_cfg:
            loss_name = loss_cfg.get("name")
            if not loss_name:
                raise ValueError("Loss configuration must include 'name' field")

            loss_params = loss_cfg.get("params", {})
            if loss_params is None:
                loss_params = {}
            if not isinstance(loss_params, dict):
                raise ValueError("Loss 'params' must be a dictionary")

            # Determine if it's a PyTorch built-in or custom loss
            if loss_name.startswith("nn."):
                loss_name = loss_name[3:]
                if not hasattr(nn, loss_name):
                    raise AttributeError(f"torch.nn has no attribute '{loss_name}'")
                loss_class = getattr(nn, loss_name)(**loss_params)
            else:
                if not hasattr(loss, loss_name):
                    raise AttributeError(f"loss module has no attribute '{loss_name}'")
                loss_class = getattr(loss, loss_name)(**loss_params)

            criteria[loss_name] = loss_class 

        return criteria


    def _train_epoch(self):
        """Train for one epoch."""
        self.model.train() # set model into training mode
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:

            # move data to 'device' and convert to float32 (model expects float32, not float64)
            inputs = batch['final'].to(self.device, dtype=torch.float32)  # (batch_size, channels, 1, time_steps) or (batch_size, channels, height, width) for images
            labels = batch['label'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # Model should output (batch_size, num_classes)

            # Compute base loss first
            base_loss, base_loss_components = self._compute_loss(outputs, labels)

            # Add regularization for backprop
            loss = base_loss
            if self.l1_reg > 0.0:
                loss += self.l1_reg * sum(torch.sum(torch.abs(p)) for p in self.model.parameters())

            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            
            # Update scheduler per batch for OneCycleLR (required for 1cycle policy)
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()  # OneCycleLR must be called per batch

            # Statistics
            total_loss += base_loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            loss_info = {f'{name}': f'{val.item():.4f}' for name, val in base_loss_components.items()}
            loss_info['total_loss'] = f'{base_loss.item():.4f}'
            loss_info['acc'] = f'{100 * correct / total:.2f}%'
            pbar.set_postfix(loss_info)

        if len(self.train_loader) == 0:
            LOGGER.warning("Training loader is empty!")
            return 0.0, 0.0
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    def _compute_loss(self, outputs, labels) -> tuple:
        loss = torch.tensor(0.0, device=self.device)
        loss_components = {}
        for idx, (loss_name, criterion) in enumerate(self.criteria.items()):
            loss_value = criterion(outputs, labels)
            loss_components[loss_name] = loss_value
            loss += loss_value * self.criteria_weights[idx]
        
        return loss, loss_components


    def _validate_epoch(self):
        """Validate an epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.val_loader, desc='Validating')
        with torch.no_grad():  # Disable gradient computation during validation
            for batch in pbar:

                inputs = batch['final'].to(self.device, dtype=torch.float32)
                labels = batch['label'].to(self.device)

                outputs = self.model(inputs)

                base_loss, base_loss_components = self._compute_loss(outputs, labels)

                # Statistics
                total_loss += base_loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Update progress bar
                loss_info = {f'{name}': f'{val.item():.4f}' for name, val in base_loss_components.items()}
                loss_info['total_loss'] = f'{base_loss.item():.4f}'
                loss_info['acc'] = f'{100 * correct / total:.2f}%'
                pbar.set_postfix(loss_info)

        if len(self.val_loader) == 0:
            LOGGER.warning("Validation loader is empty!")
            return 0.0, 0.0
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * correct / total if total > 0 else 0.0
        return avg_loss, accuracy


    def train(self):
        # Log hyperparameters to MLflow if enabled
        for logger in self.loggers:
            if isinstance(logger, MLflowLogger):
                logger.log_params({
                    'epochs': self.epochs,
                    'learning_rate': self.optimizer_cfg.get('lr'),
                    'optimizer': self.optimizer_cfg.get('name'),
                    'l1_reg': self.l1_reg,
                    'max_grad_norm': self.max_grad_norm,
                })
        
        # Log model graph to TensorBoard if enabled
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger) and logger.log_graph:
                # Try to infer input shape from first batch
                try:
                    sample_batch = next(iter(self.train_loader))
                    sample_input = sample_batch['final']
                    input_shape = sample_input.shape[1:]  # Remove batch dimension
                    logger.log_model_graph(self.model, input_shape)
                except Exception as e:
                    LOGGER.warning(f"Could not log model graph: {e}")
        
        for epoch_idx in range(self.epochs):
            LOGGER.info(f"Epoch {epoch_idx + 1}/{self.epochs}")
            
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate_epoch()
            
            metrics = {
                'train/loss': train_loss,
                'train/acc': train_acc,
                'val/loss': val_loss,
                'val/acc': val_acc
            }
            
            # Log metrics
            LOGGER.info(
                f"Epoch {epoch_idx + 1} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
            )
            
            # Log to TensorBoard and MLflow
            for logger in self.loggers:
                if isinstance(logger, TensorBoardLogger):
                    logger.log_metrics(metrics, epoch_idx)
                    if logger.log_histograms:
                        logger.log_histograms(self.model, epoch_idx)
                elif isinstance(logger, MLflowLogger):
                    logger.log_metrics(metrics, epoch_idx)

            # Checkpointing
            if self.checkpoint_manager is not None:
                self.checkpoint_manager(self.model, self.optimizer, self.scheduler, epoch_idx, metrics)

            # Early stopping
            if self.early_stopper is not None:
                current_metric = metrics[self.early_stopper.monitor]
                should_stop = self.early_stopper(current_metric, epoch_idx)
                
                if should_stop:
                    LOGGER.warning(
                        f"Early stopping triggered! "
                        f"Best {self.early_stopper.monitor}: {self.early_stopper.best_metric:.4f} "
                        f"at epoch {self.early_stopper.best_epoch + 1}"
                    )
                    break
                elif self.early_stopper.patience_counter > 0:
                    LOGGER.debug(
                        f"No improvement for {self.early_stopper.patience_counter}/{self.early_stopper.patience} epochs. "
                        f"Current {self.early_stopper.monitor}: {current_metric:.4f}, "
                        f"Best: {self.early_stopper.best_metric:.4f}"
                    )
            
            # Update scheduler per epoch (for schedulers other than OneCycleLR)
            # Note: OneCycleLR is already updated per batch in _train_epoch
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                    # Only step if not OneCycleLR (already stepped per batch)
                    self.scheduler.step()
        
        LOGGER.info("Training completed!")
        if self.early_stopper is not None and self.early_stopper.stopped:
            LOGGER.info(
                f"Best {self.early_stopper.monitor}: {self.early_stopper.best_metric:.4f} "
                f"at epoch {self.early_stopper.best_epoch + 1}"
            )
        
        # Close loggers
        for logger in self.loggers:
            if isinstance(logger, TensorBoardLogger):
                logger.close()
            elif isinstance(logger, MLflowLogger):
                if logger.should_log_model:
                    logger.log_model(self.model)
                if logger.log_artifacts and self.checkpoint_manager is not None:
                    logger.log_artifacts_dir(str(self.checkpoint_manager.save_dir), "checkpoints")
                logger.close()

    def val(self):
        pass
    

