# Experiment Tracking

## The idea

Experiment tracking means recording what you ran and what happened. Start with a Markdown table or CSV before adopting heavier tools.

## Why it matters

Research code changes quickly. If you do not record dataset version, seed, architecture, optimizer, learning rate, batch size, epochs, metrics, and notes, good results become hard to reproduce.

## Mental model

```text
configuration + code version + data version + metric = interpretable result
```

## PyTorch example

```python
run = {
    "seed": 42,
    "model": "shallow_mlp",
    "optimizer": "Adam",
    "lr": 1e-3,
    "batch_size": 32,
    "epochs": 100,
}
```

## Research-style example

```python
import csv
from pathlib import Path

def append_result(path, row):
    file_exists = Path(path).exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

append_result("experiments.csv", {
    **run,
    "best_val_loss": best_val_loss,
    "test_mse": test_mse,
    "notes": "baseline MLP",
})
```

## Common mistakes

- [ ] Recording metrics but not hyperparameters.
- [ ] Forgetting the random seed.
- [ ] Overwriting old results.
- [ ] Keeping notes only in console output.

## Previous / Next

Previous: [[02_Baseline_Model_First]]
Next: [[04_Hyperparameter_Tuning]]
Related: [[Experiment_Checklist]], [[07_Reproducible_Report]]
