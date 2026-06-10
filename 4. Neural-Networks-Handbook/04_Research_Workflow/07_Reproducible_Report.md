# Reproducible Report

## The idea

A reproducible report explains what was trained, on what data, with what settings, and what happened.

## Why it matters

Research work is easier to review, extend, and trust when someone else can recreate the main result from the report and code.

## Mental model

```text
data + model + training details + metrics + limitations + code pointers
```

## PyTorch example

```python
report_summary = {
    "dataset": "dataset_v1.csv",
    "task": "regression",
    "model": "TabularRegressor hidden=64",
    "loss": "MSELoss",
    "optimizer": "Adam lr=1e-3",
    "best_val_loss": best_val_loss,
    "test_mse": test_mse,
}
```

## Research-style example

```python
def reproducibility_block(config, metrics):
    lines = [
        "## Reproducibility",
        f"- Seed: {config['seed']}",
        f"- Dataset version: {config['dataset_version']}",
        f"- Model: {config['model_name']}",
        f"- Optimizer: {config['optimizer']} lr={config['lr']}",
        f"- Test metric: {metrics}",
    ]
    return "\n".join(lines)
```

## Common mistakes

- [ ] Omitting the dataset split rule.
- [ ] Reporting metrics without plots or error analysis.
- [ ] Hiding failed experiments that explain design choices.
- [ ] Forgetting limitations and assumptions.

## Previous / Next

Previous: [[06_Error_Analysis]]
Next: [[Regression_Project_Template]]
Related: [[Experiment_Checklist]], [[03_Experiment_Tracking]]

