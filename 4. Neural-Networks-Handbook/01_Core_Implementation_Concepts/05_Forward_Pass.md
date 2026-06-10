# Forward Pass

## The idea

The forward pass is the prediction computation. In normal PyTorch code, call `model(X)`, not `model.forward(X)`, because `model(X)` lets PyTorch run module hooks and bookkeeping.

## Why it matters

The forward pass does not update parameters. It only computes outputs from the current parameter values. Updates happen later after loss, backpropagation, and the optimizer step.

## Mental model

Shape flow for a tabular regression model:

```text
X: [32, 5]
Linear(5, 16): [32, 16]
ReLU: [32, 16]
Linear(16, 1): [32, 1]
```

## PyTorch example

```python
import torch
from torch import nn

model = nn.Sequential(
    nn.Linear(5, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
)

X = torch.randn(32, 5)
preds = model(X)
print(preds.shape)  # torch.Size([32, 1])
```

## Research-style example

```python
def trace_forward_shapes(model, X):
    out = X
    for layer in model:
        out = layer(out)
        print(type(layer).__name__, tuple(out.shape))
    return out
```

## Common mistakes

- [ ] Calling `model.forward(X)` directly in ordinary training code.
- [ ] Expecting `preds` to be probabilities when the model returns logits.
- [ ] Ignoring a final output shape that does not match the loss function.
- [ ] Updating parameters inside `forward`.

## Previous / Next

Previous: [[04_nnModule_And_Model_Classes]]
Next: [[06_Loss_Functions]]
Related: [[02_Tensors_Data_And_Shapes]], [[03_Binary_Classification_Architectures]]

