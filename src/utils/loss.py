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
    2. For each sample, summing probabilities of the true class and its adjacent classes (y-1, y, y+1)
    3. Computing negative log-likelihood of this sum
    
    This encourages the model to concentrate probability mass around the true class
    and its neighbors, which is appropriate for ordinal classification.
    
    Args:
        num_classes: Number of ordinal classes (default: 7)
        eps: Epsilon value for numerical stability (default: 1e-10)
    
    Example:
        >>> loss_fn = POM1bLoss(num_classes=7)
        >>> logits = torch.randn(32, 7)  # [batch_size, num_classes]
        >>> targets = torch.randint(0, 7, (32,))  # [batch_size]
        >>> loss = loss_fn(logits, targets)
    """
    
    def __init__(self, num_classes=7, eps=1e-10):
        super(POM1bLoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps
    
    def forward(self, logits, targets):
        """
        Compute POM1b ordinal loss.
        
        POM1b loss formula:
        L = - sum_{i} sum_{l in {-1,0,1}} log P_i(k-l)
        where k = targets[i] and P_i is the probability distribution for sample i
        
        Args:
            logits: Network output before softmax [batch_size, num_classes]
            targets: Ground truth class labels [batch_size] with values 0 to num_classes-1
        
        Returns:
            loss: Scalar loss value
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # [batch_size, num_classes]
        batch_size = logits.size(0)
        
        loss = 0.0
        for i in range(batch_size):
            k = targets[i].item()
            
            # Collect log P_i(k-l) for l in {-1,0,1} within bounds
            log_terms = []
            for l in (-1, 0, 1):
                cls = k - l
                if 0 <= cls < self.num_classes:
                    log_terms.append(torch.log(probs[i, cls] + self.eps))
            
            # Sum log terms: -sum(log(P)) = -log(prod(P))
            loss += -sum(log_terms)
        
        return loss / batch_size


# Alias for convenience
POM1b = POM1bLoss
