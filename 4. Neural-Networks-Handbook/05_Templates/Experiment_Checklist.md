# Experiment Checklist

## Before training

- [ ] Define task type: regression, binary classification, or multiclass classification.
- [ ] Confirm `X` shape and dtype.
- [ ] Confirm `y` shape and dtype.
- [ ] Create train/validation/test split.
- [ ] Fit normalization only on training data.
- [ ] Choose a baseline model.
- [ ] Record seed, dataset version, and config.

## During training

- [ ] Confirm training loss decreases on a tiny batch.
- [ ] Track train and validation loss.
- [ ] Save the best validation checkpoint.
- [ ] Record learning rate, batch size, optimizer, and scheduler.
- [ ] Watch for overfitting or underfitting.

## After training

- [ ] Load the best checkpoint.
- [ ] Evaluate on validation or test data as appropriate.
- [ ] Plot prediction vs target or confusion matrix.
- [ ] Inspect high-error or misclassified examples.
- [ ] Compare against the baseline under the same split.

## Before reporting

- [ ] Include dataset description and split rule.
- [ ] Include model architecture and parameter count.
- [ ] Include optimizer, learning rate, epochs, and batch size.
- [ ] Include final metrics and plots.
- [ ] State limitations and assumptions.
- [ ] Make the run reproducible from notes and code.

Related: [[03_Experiment_Tracking]], [[07_Reproducible_Report]]

