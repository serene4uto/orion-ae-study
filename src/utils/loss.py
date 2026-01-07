"""
Custom loss functions for Orion AE Study.

Loss functions can be used in training configuration:
  criterion:
    - name: "POM1bLoss"
      weight: 1.0
      params:
        num_classes: 7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class POM1bLoss(nn.Module):
    """
    POM1b Ordinal Loss Implementation
    
    This loss is designed for ordinal classification problems where classes have
    a natural order (e.g., tightening levels 0-6). It penalizes misclassifications
    less when they occur to adjacent classes, making it suitable for ordinal tasks.
    
    The loss works by:
    1. Computing softmax probabilities from logits
    2. For each sample, summing the LOG probabilities of the true class and its adjacent classes (k-1, k, k+1)
    3. Negating this sum
    
    Mathematically: L = -[log P(k-1) + log P(k) + log P(k+1)]
    
    This encourages the model to assign high probability to the true class
    and its neighbors, appropriate for ordinal classification.
    
    Args:
        num_classes: Number of ordinal classes (default: 7)
    
    Example:
        >>> loss_fn = POM1bLoss(num_classes=7)
        >>> logits = torch.randn(32, 7)  # [batch_size, num_classes]
        >>> targets = torch.randint(0, 7, (32,))  # [batch_size]
        >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(self, num_classes=7):
        super(POM1bLoss, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, logits, targets):
        """
        Compute POM1b ordinal loss (vectorized implementation).
        
        POM1b loss formula:
        L = - sum_{i} sum_{l in {-1,0,1}} log P_i(k-l)
        where k = targets[i] and P_i is the probability distribution for sample i
        
        Args:
            logits: Network output before softmax [batch_size, num_classes]
            targets: Ground truth class labels [batch_size] with values 0 to num_classes-1
        
        Returns:
            loss: Scalar loss value
        """
        # Use log_softmax for numerical stability (avoids log(softmax()) which can underflow)
        log_probs = F.log_softmax(logits, dim=1)  # [batch_size, num_classes]
        
        # Prepare indices: k, k-1, k+1 (clamped to valid range for gather)
        k = targets.unsqueeze(1)  # [batch_size, 1]
        k_minus_1 = (k - 1).clamp(0, self.num_classes - 1)
        k_plus_1 = (k + 1).clamp(0, self.num_classes - 1)
        
        # Gather log probabilities for each class
        log_p_k = log_probs.gather(1, k).squeeze(1)  # [batch_size]
        log_p_k_minus_1 = log_probs.gather(1, k_minus_1).squeeze(1)  # [batch_size]
        log_p_k_plus_1 = log_probs.gather(1, k_plus_1).squeeze(1)  # [batch_size]
        
        # Masks for valid neighbors (boundary handling)
        mask_k_minus_1 = (targets > 0).float()  # k-1 valid when k > 0
        mask_k_plus_1 = (targets < self.num_classes - 1).float()  # k+1 valid when k < K-1
        
        # Sum log probabilities: always include k, conditionally include neighbors
        loss_per_sample = -(log_p_k + 
                           mask_k_minus_1 * log_p_k_minus_1 + 
                           mask_k_plus_1 * log_p_k_plus_1)
        
        return loss_per_sample.mean()


# Alias for convenience
POM1b = POM1bLoss
