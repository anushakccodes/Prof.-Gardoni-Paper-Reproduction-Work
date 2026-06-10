# Debugging Training

## The idea

Training debugging means checking the simple failure points before changing the architecture. Shapes, dtypes, labels, train/eval mode, and learning rate explain many failures.

## Why it matters

A model that does not learn may have a code problem, a data problem, or a modeling problem. Debugging separates those possibilities.

## Mental model

```text
first prove the pipeline can learn something small
then scale back to the real experiment
```

## PyTorch example

```python
X_small, y_small = next(iter(train_loader))
X_small = X_small.to(device)
y_small = y_small.to(device)

for step in range(200):
    model.train()
    optimizer.zero_grad()
    loss = loss_fn(model(X_small), y_small)
    loss.backward()
    optimizer.step()
print("tiny-batch loss:", loss)
```

## Research-style example

```python
def check_batch(X, y, task):
    print("X", X.shape, X.dtype, "finite:", torch.isfinite(X).all().item())
    print("y", y.shape, y.dtype)
    if task == "multiclass":
        print("classes:", torch.unique(y))

def check_gradients(model):
    for name, p in model.named_parameters():
        if p.grad is None:
            print(name, "no grad")
        else:
            print(name, p.grad.norm().item())
```

## Common mistakes

- [ ] Not checking shapes and dtypes first.
- [ ] Skipping the tiny-batch overfit test.
- [ ] Forgetting to inspect labels.
- [ ] Debugging validation metrics before confirming training loss can decrease.
- [ ] Ignoring data leakage when results look too good.

## Previous / Next

Previous: [[06_Skip_Connections]]
Next: [[01_From_Problem_To_Dataset]]
Related: [[02_Tensors_Data_And_Shapes]], [[09_Training_Loop]]
