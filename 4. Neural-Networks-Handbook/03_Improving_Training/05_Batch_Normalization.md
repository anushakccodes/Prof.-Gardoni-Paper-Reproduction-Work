# Batch Normalization

## The idea

Batch normalization normalizes intermediate activations using batch statistics during training and stored running statistics during evaluation.

## Why it matters

It can stabilize training, but it also adds train/eval behavior that must be handled correctly. Very small batch sizes can make batch statistics noisy.

## Mental model

```text
Linear -> BatchNorm1d -> ReLU
```

## PyTorch example

```python
from torch import nn

model = nn.Sequential(
    nn.Linear(12, 64),
    nn.BatchNorm1d(64),
    nn.ReLU(),
    nn.Linear(64, 1),
)
```

## Research-style example

```python
class BatchNormMLP(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, X):
        return self.net(X)
```

## Common mistakes

- [ ] Forgetting that batch normalization behaves differently in train and eval modes.
- [ ] Using tiny batches and trusting noisy batch statistics.
- [ ] Placing batch normalization after the final output without a clear reason.
- [ ] Comparing models without recording whether batch normalization was used.

## Previous / Next

Previous: [[04_Dropout]]
Next: [[06_Skip_Connections]]
Related: [[09_Training_Loop]], [[07_Debugging_Training]]

