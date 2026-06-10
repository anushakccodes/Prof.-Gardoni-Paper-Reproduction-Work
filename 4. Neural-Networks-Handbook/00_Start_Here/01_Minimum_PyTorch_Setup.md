# Minimum PyTorch Setup

## The idea

Use a small Python environment where PyTorch, NumPy, pandas, matplotlib, and scikit-learn are installed. Keep the setup boring so training code is easier to debug.

## Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch numpy pandas matplotlib scikit-learn
```

For GPU-specific installation commands, use the selector on the PyTorch installation page. ([PyTorch docs](https://docs.pytorch.org/get-started/locally/))

## Check the install

```python
import torch

print(torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device:", "cuda" if torch.cuda.is_available() else "cpu")
```

## Imports used in this handbook

```python
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
```

## Seed helper

```python
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

## Device pattern

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

Previous: [[00_Neural_Networks_Handbook]]
Next: [[01_The_Neural_Network_Loop]]
Related: [[PyTorch_Glossary]]

