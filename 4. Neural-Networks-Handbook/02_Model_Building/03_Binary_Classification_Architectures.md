# Binary Classification Architectures

## The idea

A binary classifier predicts one output logit per example. A logit is a raw score. `BCEWithLogitsLoss` combines sigmoid and binary cross-entropy in a numerically stable way.

## Why it matters

The training output and the evaluation output are different views of the same model. Train with logits. Convert logits to probabilities only when interpreting or computing probability-based metrics.

## Mental model

```text
X: [B, D] -> logit: [B, 1] -> sigmoid(logit): [B, 1] probability
```

## PyTorch example

```python
import torch
from torch import nn

model = nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1))
loss_fn = nn.BCEWithLogitsLoss()

X = torch.randn(10, 4)
y = torch.randint(0, 2, (10, 1)).float()
loss = loss_fn(model(X), y)
```

## Research-style example

```python
def predict_binary(model, X, threshold=0.5):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.sigmoid(logits)
        labels = (probs >= threshold).long()
    return probs, labels
```

## Common mistakes

- [ ] Using two output units for a simple binary classifier without a reason.
- [ ] Passing integer targets into `BCEWithLogitsLoss`.
- [ ] Applying sigmoid before the loss.
- [ ] Keeping the default `0.5` threshold without checking precision and recall.

## Previous / Next

Previous: [[02_Regression_Architectures]]
Next: [[04_Multiclass_Classification_Architectures]]
Related: [[06_Loss_Functions]], [[11_Evaluation_Metrics]]

