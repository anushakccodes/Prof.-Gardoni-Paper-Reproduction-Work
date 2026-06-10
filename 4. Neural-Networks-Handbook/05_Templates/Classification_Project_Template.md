# Classification Project Template

Use one output logit with `BCEWithLogitsLoss` for binary classification. Use `num_classes` output logits with `CrossEntropyLoss` for multiclass classification.

```python
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, confusion_matrix

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
task = "multiclass"  # "binary" or "multiclass"

X = torch.randn(600, 10, dtype=torch.float32)

if task == "binary":
    y = (X[:, 0] + X[:, 1] > 0).float().view(-1, 1)
    output_dim = 1
    loss_fn = nn.BCEWithLogitsLoss()
else:
    y = torch.randint(0, 3, (600,), dtype=torch.long)
    output_dim = 3
    loss_fn = nn.CrossEntropyLoss()

dataset = TensorDataset(X, y)
train_ds, val_ds, test_ds = random_split(dataset, [420, 90, 90], generator=torch.Generator().manual_seed(42))
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

class Classifier(nn.Module):
    def __init__(self, num_features, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, X):
        return self.net(X)

model = Classifier(X.shape[1], output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch():
    model.train()
    total = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)
        loss.backward()
        optimizer.step()
        total += loss.item() * X_batch.size(0)
    return total / len(train_loader.dataset)

for epoch in range(50):
    train_one_epoch()

all_logits, all_targets = [], []
model.eval()
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        all_logits.append(model(X_batch.to(device)).cpu())
        all_targets.append(y_batch)

logits = torch.cat(all_logits)
y_true = torch.cat(all_targets).numpy()

if task == "binary":
    y_pred = (torch.sigmoid(logits).view(-1) >= 0.5).long().numpy()
else:
    y_pred = torch.argmax(logits, dim=1).numpy()

print(classification_report(y_true, y_pred, zero_division=0))
print(confusion_matrix(y_true, y_pred))
```

Related: [[03_Binary_Classification_Architectures]], [[04_Multiclass_Classification_Architectures]]

