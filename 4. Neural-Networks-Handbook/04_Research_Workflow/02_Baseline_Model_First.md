# Baseline Model First

## The idea

A baseline is a simple model or rule used as a comparison point. Start with a dummy predictor, a linear/logistic model, or a shallow neural network before adding complexity.

## Why it matters

Without a baseline, you cannot tell whether a neural network is actually useful. A baseline also catches data and metric mistakes early.

## Mental model

```text
simple model beats guessing
better model should beat simple model
complex model must justify its complexity
```

## PyTorch example

```python
from torch import nn

baseline = nn.Linear(num_features, 1)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(baseline.parameters(), lr=1e-3)
```

## Research-style example

```python
results = []
for name, model in {
    "linear": nn.Linear(num_features, 1),
    "shallow_mlp": nn.Sequential(nn.Linear(num_features, 32), nn.ReLU(), nn.Linear(32, 1)),
}.items():
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(50):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss = validate_one_epoch(model, val_loader, loss_fn, device)
    results.append({"model": name, "val_loss": val_loss})
```

## Common mistakes

- [ ] Skipping the baseline because the final model seems obvious.
- [ ] Comparing a tuned neural network to an untuned baseline.
- [ ] Using a different train/validation split for each model.
- [ ] Adding layers before checking whether a linear model is already strong.

## Previous / Next

Previous: [[01_From_Problem_To_Dataset]]
Next: [[03_Experiment_Tracking]]
Related: [[05_Model_Comparison]], [[02_Regression_Architectures]]

