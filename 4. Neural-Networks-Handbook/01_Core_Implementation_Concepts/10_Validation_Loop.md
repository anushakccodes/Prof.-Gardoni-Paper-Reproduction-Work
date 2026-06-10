# Validation Loop

## The idea

Validation estimates how well the model performs on data not used for parameter updates. It uses `model.eval()` and `torch.no_grad()` so layers behave correctly and gradients are not stored.

## Why it matters

Validation helps choose checkpoints, tune hyperparameters, and detect overfitting. It should not leak information into training.

## Mental model

```text
training loss asks: how well did we fit update data?
validation loss asks: how well might we generalize?
```

## PyTorch example

```python
def validate_one_epoch(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            preds = model(X_batch)
            loss = loss_fn(preds, y_batch)
            total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(val_loader.dataset)
```

## Research-style example

```python
best_val_loss = float("inf")

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_model.pt")
```

## Common mistakes

- [ ] Using validation data in normalization fitting.
- [ ] Calling `loss.backward()` in validation.
- [ ] Forgetting `model.eval()` before validation.
- [ ] Tuning repeatedly on the test set instead of validation data.

## Previous / Next

Previous: [[09_Training_Loop]]
Next: [[11_Evaluation_Metrics]]
Related: [[02_Overfitting_Underfitting]], [[06_Saving_Loading_Checkpoints]]

