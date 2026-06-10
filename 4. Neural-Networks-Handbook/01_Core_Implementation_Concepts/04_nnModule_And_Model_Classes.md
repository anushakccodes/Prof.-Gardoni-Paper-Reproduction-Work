# nn.Module and Model Classes

## The idea

In PyTorch, custom neural networks usually inherit from `nn.Module`, the base class for neural-network modules. ([PyTorch docs](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html)) Layers are defined in `__init__`, and the prediction computation is defined in `forward`.

## Why it matters

When layers are stored as attributes, PyTorch can find their parameters through `model.parameters()`. That is what lets the optimizer update the model.

## Mental model

```text
__init__ -> what layers exist
forward  -> how data moves through those layers
```

## PyTorch example

```python
import torch
from torch import nn

class SmallRegressor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, X):
        return self.net(X)

model = SmallRegressor(num_features=5)
print(sum(p.numel() for p in model.parameters()))
```

## Research-style example

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SmallRegressor(num_features=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## Common mistakes

- [ ] Forgetting `super().__init__()`.
- [ ] Creating layers inside `forward`, which prevents stable parameter tracking.
- [ ] Passing `model.parameters` instead of `model.parameters()`.
- [ ] Moving data to GPU but leaving the model on CPU.

## Previous / Next

Previous: [[03_Dataset_Dataloader_Batch_Epoch]]
Next: [[05_Forward_Pass]]
Related: [[08_Optimizers]], [[02_Regression_Architectures]]

