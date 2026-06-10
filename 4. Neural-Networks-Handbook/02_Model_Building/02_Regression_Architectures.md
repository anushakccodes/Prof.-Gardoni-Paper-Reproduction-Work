# Regression Architectures

## The idea

A regression model predicts continuous values. For tabular regression, the model usually maps `[batch_size, num_features]` to `[batch_size, num_outputs]`.

## Why it matters

Ordinary regression usually has no final activation. The model should be free to predict negative, positive, small, or large values unless the target requires a constraint.

## Mental model

```text
features -> hidden layers -> continuous output
X: [B, D] -> y_hat: [B, 1]
```

## PyTorch example

```python
from torch import nn

model = nn.Sequential(
    nn.Linear(6, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
)
loss_fn = nn.MSELoss()
```

## Research-style example

```python
class TabularRegressor(nn.Module):
    def __init__(self, num_features, hidden=64, num_outputs=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_outputs),
        )

    def forward(self, X):
        return self.net(X)
```

## Common mistakes

- [ ] Adding sigmoid to the final layer for unconstrained regression.
- [ ] Using classification labels with regression loss.
- [ ] Forgetting to scale features when their ranges differ greatly.
- [ ] Reporting only MSE when MAE or R2 is easier to interpret.

## Previous / Next

Previous: [[01_Activation_Functions]]
Next: [[03_Binary_Classification_Architectures]]
Related: [[Regression_Project_Template]], [[11_Evaluation_Metrics]]

