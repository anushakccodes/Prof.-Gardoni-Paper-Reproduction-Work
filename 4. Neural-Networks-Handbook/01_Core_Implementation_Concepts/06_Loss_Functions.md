# Loss Functions

## The idea

A loss function measures how wrong the model is for one batch. PyTorch provides common losses in `torch.nn`. ([PyTorch docs](https://docs.pytorch.org/docs/stable/nn.html))

Use `MSELoss` for ordinary regression, `BCEWithLogitsLoss` for binary classification with one logit, and `CrossEntropyLoss` for multiclass classification with class-index labels.

## Why it matters

The loss defines what the model is trying to optimize. A wrong loss or target shape can make training meaningless even if the code runs.

## Mental model

```text
regression output: [B, 1], target: [B, 1]
binary output logits: [B, 1], target: [B, 1] float
multiclass output logits: [B, C], target: [B] long
```

## PyTorch example

```python
import torch
from torch import nn

mse = nn.MSELoss()
bce = nn.BCEWithLogitsLoss()
ce = nn.CrossEntropyLoss()

reg_loss = mse(torch.randn(8, 1), torch.randn(8, 1))
bin_loss = bce(torch.randn(8, 1), torch.randint(0, 2, (8, 1)).float())
multi_loss = ce(torch.randn(8, 3), torch.randint(0, 3, (8,)))
```

## Research-style example

```python
def choose_loss(task):
    if task == "regression":
        return nn.MSELoss()
    if task == "binary":
        return nn.BCEWithLogitsLoss()
    if task == "multiclass":
        return nn.CrossEntropyLoss()
    raise ValueError(f"unknown task: {task}")
```

## Common mistakes

- [ ] Applying sigmoid before `BCEWithLogitsLoss`.
- [ ] Applying softmax before `CrossEntropyLoss`.
- [ ] Using one-hot labels with `CrossEntropyLoss`.
- [ ] Comparing `[B]` predictions with `[B, 1]` targets accidentally.

## Previous / Next

Previous: [[05_Forward_Pass]]
Next: [[07_Backpropagation_Autograd]]
Related: [[11_Evaluation_Metrics]], [[04_Multiclass_Classification_Architectures]]

