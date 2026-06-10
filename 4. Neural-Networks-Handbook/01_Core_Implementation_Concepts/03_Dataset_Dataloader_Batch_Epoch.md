# Dataset, DataLoader, Batch, and Epoch

## The idea

PyTorch separates data access from model training through `Dataset` and `DataLoader`. ([PyTorch docs](https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html)) A `Dataset` knows how to return one example. A `DataLoader` groups examples into batches and makes them iterable.

An epoch means one pass through the training dataset.

## Why it matters

Keeping data code separate from model code makes experiments easier to inspect. You can change the model without rewriting CSV parsing, and you can test the dataset before training.

## Mental model

```text
Dataset[index] -> one (X, y) pair
DataLoader -> many batches
epoch -> loop over all batches once
```

## PyTorch example

```python
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

X = torch.randn(100, 5)
y = torch.randn(100, 1)
dataset = TensorDataset(X, y)

train_ds, val_ds = random_split(dataset, [80, 20])
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

for X_batch, y_batch in train_loader:
    print(X_batch.shape, y_batch.shape)
    break
```

## Research-style example

```python
import pandas as pd
import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, path, feature_cols, target_col):
        df = pd.read_csv(path)
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[[target_col]].values, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
```

## Common mistakes

- [ ] Shuffling validation or test data when order matters for analysis.
- [ ] Normalizing validation data using statistics from validation itself.
- [ ] Mixing data loading logic into the model class.
- [ ] Treating one batch as one epoch.

## Previous / Next

Previous: [[02_Tensors_Data_And_Shapes]]
Next: [[04_nnModule_And_Model_Classes]]
Related: [[Custom_Dataset_Template]], [[01_From_Problem_To_Dataset]]

