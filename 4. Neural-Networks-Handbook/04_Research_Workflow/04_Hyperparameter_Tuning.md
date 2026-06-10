# Hyperparameter Tuning

## The idea

Hyperparameters are choices set before or around training: learning rate, batch size, hidden layers, hidden units, dropout, weight decay, scheduler, and number of epochs.

## Why it matters

Changing hyperparameters can affect performance as much as changing architecture. Tune on validation data, not test data.

## Mental model

```text
train set -> fit parameters
validation set -> choose hyperparameters
test set -> final estimate
```

## PyTorch example

```python
configs = [
    {"lr": 1e-2, "hidden": 32, "weight_decay": 0.0},
    {"lr": 1e-3, "hidden": 64, "weight_decay": 1e-4},
]
```

## Research-style example

```python
def run_config(config):
    model = TabularRegressor(num_features, hidden=config["hidden"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    best_val = float("inf")
    for epoch in range(50):
        train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, loss_fn, device)
        best_val = min(best_val, val_loss)
    return best_val
```

## Common mistakes

- [ ] Tuning on the test set.
- [ ] Changing many hyperparameters at once without tracking them.
- [ ] Searching only architectures while ignoring learning rate.
- [ ] Keeping the best-looking run without rerunning it with a fixed seed.

## Previous / Next

Previous: [[03_Experiment_Tracking]]
Next: [[05_Model_Comparison]]
Related: [[01_Learning_Rate_And_Schedulers]], [[03_Regularization]]

