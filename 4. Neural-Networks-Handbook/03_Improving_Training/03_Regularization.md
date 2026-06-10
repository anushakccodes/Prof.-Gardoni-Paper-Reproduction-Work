# Regularization

## The idea

Regularization means adding constraints or habits that reduce overfitting. Common first tools are weight decay, early stopping, simpler models, and sometimes data augmentation.

## Why it matters

Regularization helps a model generalize instead of memorizing. It is especially useful when the training set is small relative to model capacity.

## Mental model

```text
fit training data well
but avoid solutions that are unnecessarily complex
```

## PyTorch example

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)
```

## Research-style example

```python
best_val = float("inf")
patience = 10
epochs_without_improvement = 0

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
    val_loss = validate_one_epoch(model, val_loader, loss_fn, device)

    if val_loss < best_val:
        best_val = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pt")
    else:
        epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
```

## Common mistakes

- [ ] Adding regularization before confirming the training loop works.
- [ ] Using too much weight decay and causing underfitting.
- [ ] Calling early stopping based on noisy one-epoch changes.
- [ ] Treating data augmentation as automatically valid for tabular data.

## Previous / Next

Previous: [[02_Overfitting_Underfitting]]
Next: [[04_Dropout]]
Related: [[04_Hyperparameter_Tuning]], [[06_Saving_Loading_Checkpoints]]

