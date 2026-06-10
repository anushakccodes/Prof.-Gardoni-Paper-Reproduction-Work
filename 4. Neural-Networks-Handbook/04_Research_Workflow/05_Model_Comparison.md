# Model Comparison

## The idea

Model comparison means evaluating candidate models under the same conditions: same split, same metric, same preprocessing, and preferably multiple seeds.

## Why it matters

A model is not better just because it was trained longer, tuned more carefully, or evaluated on an easier split.

## Mental model

```text
fair comparison = same data + same metric + clear complexity cost
```

## PyTorch example

```python
comparison_row = {
    "model": "mlp_2x64",
    "seed": 42,
    "val_mse": 0.128,
    "test_mse": 0.141,
    "num_parameters": sum(p.numel() for p in model.parameters()),
}
```

## Research-style example

```python
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

rows = []
for seed in [1, 2, 3]:
    set_seed(seed)
    model = TabularRegressor(num_features, hidden=64).to(device)
    rows.append({
        "model": "mlp_2x64",
        "seed": seed,
        "params": count_parameters(model),
        "best_val_loss": train_and_select(model),
    })
```

## Common mistakes

- [ ] Comparing models on different splits.
- [ ] Reporting only the best seed.
- [ ] Ignoring parameter count and training time.
- [ ] Changing preprocessing between models.

## Previous / Next

Previous: [[04_Hyperparameter_Tuning]]
Next: [[06_Error_Analysis]]
Related: [[02_Baseline_Model_First]], [[07_Reproducible_Report]]

