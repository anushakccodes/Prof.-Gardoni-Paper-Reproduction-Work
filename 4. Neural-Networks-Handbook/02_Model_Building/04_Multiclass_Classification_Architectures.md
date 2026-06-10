# Multiclass Classification Architectures

## The idea

A multiclass classifier predicts one logit for each class. If there are `C` classes, the output shape is `[batch_size, C]`. `CrossEntropyLoss` expects raw logits and target labels shaped `[batch_size]`.

## Why it matters

The final layer size must match the number of classes. Label dtype and shape matter more here than beginners expect.

## Mental model

```text
X: [B, D] -> logits: [B, C]
y: [B] with values from 0 to C - 1
```

## PyTorch example

```python
import torch
from torch import nn

num_classes = 3
model = nn.Sequential(nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, num_classes))
loss_fn = nn.CrossEntropyLoss()

X = torch.randn(12, 5)
y = torch.randint(0, num_classes, (12,), dtype=torch.long)
loss = loss_fn(model(X), y)
```

## Research-style example

```python
def predict_multiclass(model, X):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        labels = torch.argmax(probs, dim=1)
    return probs, labels
```

## Common mistakes

- [ ] Applying softmax before `CrossEntropyLoss`.
- [ ] Using one-hot labels with `CrossEntropyLoss`.
- [ ] Setting output dimension to `1` for a multiclass problem.
- [ ] Forgetting that class labels must be `torch.long`.

## Previous / Next

Previous: [[03_Binary_Classification_Architectures]]
Next: [[01_Learning_Rate_And_Schedulers]]
Related: [[Classification_Project_Template]], [[11_Evaluation_Metrics]]

