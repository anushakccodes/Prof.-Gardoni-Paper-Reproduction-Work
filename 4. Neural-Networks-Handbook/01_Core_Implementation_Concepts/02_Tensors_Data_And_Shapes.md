# Tensors, Data, and Shapes

## The idea

A tensor is an array that PyTorch can move through model layers and differentiate when needed. A scalar has shape `[]`, a vector might be `[D]`, a matrix might be `[N, D]`, and a batch of tabular examples is usually `[batch_size, num_features]`. ([PyTorch docs](https://docs.pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html))

## Why it matters

Neural-network code is mostly shape plumbing. If `X`, `y`, model outputs, and loss functions disagree about shape or dtype, training fails or silently learns the wrong thing.

## Mental model

Common supervised-learning shapes:

```text
X: [batch_size, num_features]
y regression: [batch_size, 1]
y binary classification: [batch_size, 1]
y multiclass classification: [batch_size]
```

For multiclass classification, labels are class indices like `0`, `1`, `2`, not one-hot rows unless you intentionally choose a different loss.

## PyTorch example

```python
import torch

X = torch.tensor([[1.2, 0.7, 3.1],
                  [0.4, 2.2, 1.8]], dtype=torch.float32)
y_reg = torch.tensor([[10.5], [7.2]], dtype=torch.float32)
y_class = torch.tensor([2, 0], dtype=torch.long)

print(X.shape)       # torch.Size([2, 3])
print(y_reg.shape)   # torch.Size([2, 1])
print(y_class.shape) # torch.Size([2])
```

## Research-style example

```python
def prepare_tabular_batch(features, targets, task):
    X = torch.as_tensor(features, dtype=torch.float32)
    if task == "multiclass":
        y = torch.as_tensor(targets, dtype=torch.long)
    else:
        y = torch.as_tensor(targets, dtype=torch.float32).view(-1, 1)
    return X, y
```

## Common mistakes

- [ ] Using integer tensors for input features.
- [ ] Passing `float32` labels to `CrossEntropyLoss`.
- [ ] Using `[batch_size, 1]` labels with `CrossEntropyLoss`.
- [ ] Forgetting that `view(-1, 1)` changes target shape.

## Previous / Next

Previous: [[01_The_Neural_Network_Loop]]
Next: [[03_Dataset_Dataloader_Batch_Epoch]]
Related: [[06_Loss_Functions]], [[PyTorch_Glossary]]

