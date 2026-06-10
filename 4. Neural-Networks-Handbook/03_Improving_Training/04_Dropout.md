# Dropout

## The idea

Dropout randomly sets some activations to zero during training. It is active in `model.train()` and disabled in `model.eval()`.

## Why it matters

Dropout can reduce overfitting by preventing hidden units from depending too strongly on each other. It is not always helpful for small tabular datasets, so test it.

## Mental model

```text
training: some hidden activations are dropped
evaluation: all hidden activations are used
```

## PyTorch example

```python
from torch import nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Dropout(p=0.2),
    nn.Linear(64, 1),
)
```

## Research-style example

```python
class DropoutRegressor(nn.Module):
    def __init__(self, num_features, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, X):
        return self.net(X)
```

## Common mistakes

- [ ] Forgetting `model.eval()` before validation or testing.
- [ ] Using very high dropout and causing underfitting.
- [ ] Expecting dropout to fix data leakage.
- [ ] Adding dropout after the final output layer.

## Previous / Next

Previous: [[03_Regularization]]
Next: [[05_Batch_Normalization]]
Related: [[09_Training_Loop]], [[10_Validation_Loop]]

