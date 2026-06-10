# Training Loop

## The idea

The training loop repeats the neural-network loop over many batches and epochs. A typical PyTorch training pass uses `model.train()`, forward pass, loss, backward pass, and optimizer step. ([PyTorch docs](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html))

## Why it matters

The training loop is where model behavior, data behavior, and optimizer behavior meet. Keeping it visible makes research code easier to debug.

## Mental model

```text
epoch -> many batches -> one update per batch
```

## PyTorch example

```python
def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = loss_fn(preds, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

    return total_loss / len(train_loader.dataset)
```

## Research-style example

```python
history = []
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    history.append({"epoch": epoch + 1, "train_loss": train_loss})
    print(f"epoch {epoch + 1}: train_loss={train_loss:.4f}")
```

## Common mistakes

- [ ] Averaging batch losses without accounting for the final smaller batch.
- [ ] Forgetting `model.train()` when using dropout or batch normalization.
- [ ] Not moving every tensor to the same device.
- [ ] Printing loss without storing it for later plots.

## Previous / Next

Previous: [[08_Optimizers]]
Next: [[10_Validation_Loop]]
Related: [[Training_Loop_Template]], [[07_Debugging_Training]]

