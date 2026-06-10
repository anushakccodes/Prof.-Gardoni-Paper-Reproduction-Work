# Skip Connections

## The idea

A skip connection adds an identity shortcut so a block can learn a change to the input instead of an entirely new representation.

## Why it matters

Skip connections can help gradients move through deeper models. In this handbook, keep them conceptual and use them only for simple MLP blocks.

## Mental model

```text
output = input + block(input)
```

## PyTorch example

```python
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )

    def forward(self, X):
        return X + self.block(X)
```

## Research-style example

```python
class ResidualMLP(nn.Module):
    def __init__(self, num_features, width=64):
        super().__init__()
        self.input = nn.Linear(num_features, width)
        self.hidden = ResidualBlock(width)
        self.output = nn.Linear(width, 1)

    def forward(self, X):
        X = torch.relu(self.input(X))
        X = torch.relu(self.hidden(X))
        return self.output(X)
```

## Common mistakes

- [ ] Adding tensors with different shapes.
- [ ] Making the model deeper before the baseline works.
- [ ] Treating skip connections as a substitute for debugging.
- [ ] Copying image-model patterns into tabular MLPs without a reason.

## Previous / Next

Previous: [[05_Batch_Normalization]]
Next: [[07_Debugging_Training]]
Related: [[02_Baseline_Model_First]], [[07_Backpropagation_Autograd]]

