# Backpropagation and Autograd

## The idea

PyTorch autograd is the automatic differentiation engine that computes gradients for training. ([PyTorch docs](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)) A gradient tells how a small change in a parameter would affect the loss.

## Why it matters

`loss.backward()` fills each trainable parameter's `.grad` field. The optimizer then uses those gradients to update parameters.

## Mental model

```text
loss.backward() asks:
"For every parameter that helped create this loss, which direction would reduce it?"
```

## PyTorch example

```python
import torch
from torch import nn

model = nn.Linear(3, 1)
X = torch.randn(4, 3)
y = torch.randn(4, 1)
loss_fn = nn.MSELoss()

preds = model(X)
loss = loss_fn(preds, y)
loss.backward()

print(model.weight.grad)
```

## Research-style example

```python
def gradient_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5

optimizer.zero_grad()
loss = loss_fn(model(X_batch), y_batch)
loss.backward()
print("grad norm:", gradient_norm(model))
optimizer.step()
```

## Common mistakes

- [ ] Forgetting `optimizer.zero_grad()`, which accumulates old gradients.
- [ ] Calling `backward()` during validation.
- [ ] Detaching tensors before computing the training loss.
- [ ] Expecting gradients for tensors with `requires_grad=False`.

## Previous / Next

Previous: [[06_Loss_Functions]]
Next: [[08_Optimizers]]
Related: [[10_Validation_Loop]], [[07_Debugging_Training]]

