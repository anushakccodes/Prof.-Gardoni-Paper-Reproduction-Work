# From Problem to Dataset

## The idea

Before writing a model, define the input features, the target, and whether the task is regression, binary classification, or multiclass classification.

## Why it matters

Most research implementation problems begin as data-definition problems. A clear dataset specification prevents target leakage, label confusion, and unfair evaluation.

## Mental model

```text
research question -> target y
available measurements -> features X
task type -> model output and loss
```

## PyTorch example

```python
import torch
from torch.utils.data import TensorDataset

X = torch.randn(200, 8).float()
y = torch.randn(200, 1).float()
dataset = TensorDataset(X, y)
```

## Research-style example

```python
def make_splits(dataset, train_frac=0.7, val_frac=0.15, seed=42):
    n = len(dataset)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    n_test = n - n_train - n_val
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.random_split(dataset, [n_train, n_val, n_test], generator=generator)
```

Document feature columns, target definition, split rule, normalization rule, and assumptions in the experiment log.

## Common mistakes

- [ ] Letting future information leak into features.
- [ ] Normalizing validation/test data using their own statistics.
- [ ] Changing the target definition mid-project without renaming the dataset version.
- [ ] Starting with a complex model before checking the dataset.

## Previous / Next

Previous: [[07_Debugging_Training]]
Next: [[02_Baseline_Model_First]]
Related: [[03_Dataset_Dataloader_Batch_Epoch]], [[03_Experiment_Tracking]]

