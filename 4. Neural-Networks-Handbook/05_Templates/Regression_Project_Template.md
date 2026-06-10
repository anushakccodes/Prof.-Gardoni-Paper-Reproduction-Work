# Regression Project Template

Use this as a small starting point for tabular regression.

```python
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Placeholder data: replace with real normalized features and targets.
X = torch.randn(1000, 8, dtype=torch.float32)
y = (X[:, [0]] * 2.0 - X[:, [1]] + 0.1 * torch.randn(1000, 1)).float()

dataset = TensorDataset(X, y)
train_ds, val_ds, test_ds = random_split(
    dataset, [700, 150, 150], generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)
test_loader = DataLoader(test_ds, batch_size=64)

class Regressor(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, X):
        return self.net(X)

model = Regressor(num_features=X.shape[1]).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def train_one_epoch():
    model.train()
    total = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total += loss.item() * X_batch.size(0)
    return total / len(train_loader.dataset)

def evaluate_loss(loader):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            loss = loss_fn(model(X_batch), y_batch)
            total += loss.item() * X_batch.size(0)
    return total / len(loader.dataset)

history = []
best_val = float("inf")
for epoch in range(100):
    train_loss = train_one_epoch()
    val_loss = evaluate_loss(val_loader)
    history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_regressor.pt")

model.load_state_dict(torch.load("best_regressor.pt", map_location=device))
model.eval()

preds, targets = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        preds.append(model(X_batch.to(device)).cpu())
        targets.append(y_batch)

y_pred = torch.cat(preds).numpy()
y_true = torch.cat(targets).numpy()
print("MSE:", mean_squared_error(y_true, y_pred))
print("MAE:", mean_absolute_error(y_true, y_pred))
print("R2:", r2_score(y_true, y_pred))

plt.scatter(y_true, y_pred, alpha=0.5)
plt.xlabel("Target")
plt.ylabel("Prediction")
plt.title("Prediction vs target")
plt.show()
```

Related: [[02_Regression_Architectures]], [[Evaluation_Template]]

