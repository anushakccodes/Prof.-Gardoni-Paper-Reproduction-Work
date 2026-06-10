# Optimizers

## The idea

An optimizer updates model parameters using gradients. Common first choices are SGD and Adam. PyTorch optimizers live in `torch.optim`. ([PyTorch docs](https://docs.pytorch.org/docs/stable/optim.html))

The learning rate controls the update size.

## Why it matters

Too large a learning rate can make training unstable. Too small a learning rate can make training look stuck.

## Mental model

```text
zero_grad -> backward computes gradients -> step updates parameters
```

## PyTorch example

```python
import torch
from torch import nn

model = nn.Linear(4, 1)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

optimizer.zero_grad()
loss = nn.MSELoss()(model(torch.randn(8, 4)), torch.randn(8, 1))
loss.backward()
optimizer.step()
```

## Research-style example

```python
def make_optimizer(model, name="adam", lr=1e-3, weight_decay=0.0):
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")
```

## Common mistakes

- [ ] Passing tensors instead of `model.parameters()` to the optimizer.
- [ ] Calling `step()` without a fresh backward pass.
- [ ] Using the same learning rate after changing the model or batch size.
- [ ] Forgetting that optimizer state should be saved in checkpoints.

## Previous / Next

Previous: [[07_Backpropagation_Autograd]]
Next: [[09_Training_Loop]]
Related: [[01_Learning_Rate_And_Schedulers]], [[06_Saving_Loading_Checkpoints]]

