# Evaluation Metrics

## The idea

Loss is used for optimization, while metrics are used to judge performance. For regression, MSE and R2 are common first metrics. ([MSE docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html), [R2 docs](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html))

## Why it matters

The metric should match the research question. A low training loss is not enough if the final test metric does not improve.

## Mental model

```text
train loss -> optimization signal
validation loss/metric -> model selection signal
test metric -> final held-out estimate
```

## PyTorch example

```python
import torch

logits = torch.tensor([[2.0], [-1.0], [0.7]])
probs = torch.sigmoid(logits)
pred_labels = (probs >= 0.5).long()
print(pred_labels.view(-1))
```

## Research-style example

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def regression_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

def binary_metrics(y_true, logits):
    probs = torch.sigmoid(torch.tensor(logits)).numpy().reshape(-1)
    y_pred = (probs >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    return {"accuracy": accuracy_score(y_true, y_pred), "precision": precision, "recall": recall, "f1": f1}
```

## Common mistakes

- [ ] Reporting only training loss.
- [ ] Choosing accuracy for heavily imbalanced binary data without checking recall or F1.
- [ ] Computing multiclass accuracy from probabilities before selecting class indices.
- [ ] Changing metrics between models in the same comparison.

## Previous / Next

Previous: [[10_Validation_Loop]]
Next: [[01_Activation_Functions]]
Related: [[Evaluation_Template]], [[05_Model_Comparison]]

