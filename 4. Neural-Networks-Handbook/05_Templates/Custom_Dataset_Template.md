# Custom Dataset Template

Use this when data lives in a CSV or needs custom parsing.

```python
import pandas as pd
import torch
from torch.utils.data import Dataset

class CSVDataset(Dataset):
    def __init__(self, csv_path, feature_cols, target_col, task="regression"):
        self.df = pd.read_csv(csv_path)
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.task = task

        self.X = torch.tensor(self.df[feature_cols].values, dtype=torch.float32)

        if task == "multiclass":
            self.y = torch.tensor(self.df[target_col].values, dtype=torch.long)
        else:
            self.y = torch.tensor(self.df[[target_col]].values, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Shape notes:
# regression or binary y: [batch_size, 1]
# multiclass y: [batch_size]
```

Related: [[03_Dataset_Dataloader_Batch_Epoch]], [[02_Tensors_Data_And_Shapes]]

