# Saving and Loading Checkpoints

## The idea

PyTorch models are commonly saved through `state_dict`, a dictionary of parameter tensors. Saving and loading model weights is part of the standard beginner workflow. ([PyTorch docs](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html))

## Why it matters

Research experiments need recoverable model states. A checkpoint can store weights, optimizer state, epoch number, validation loss, and configuration.

## Mental model

```text
weights only -> useful for inference
full checkpoint -> useful for resuming training
```

## PyTorch example

```python
torch.save(model.state_dict(), "model_weights.pt")

model = TabularRegressor(num_features=6)
model.load_state_dict(torch.load("model_weights.pt"))
model.eval()
```

## Research-style example

```python
checkpoint = {
    "epoch": epoch,
    "model_state": model.state_dict(),
    "optimizer_state": optimizer.state_dict(),
    "val_loss": val_loss,
    "config": {"lr": 1e-3, "hidden": 64},
}
torch.save(checkpoint, "checkpoint.pt")

loaded = torch.load("checkpoint.pt", map_location=device)
model.load_state_dict(loaded["model_state"])
optimizer.load_state_dict(loaded["optimizer_state"])
```

## Common mistakes

- [ ] Saving the whole model object when `state_dict` is enough.
- [ ] Loading weights into a model with a different architecture.
- [ ] Forgetting `map_location` when loading across CPU/GPU.
- [ ] Not saving the validation metric that selected the checkpoint.

## Previous / Next

Previous: [[05_Weight_Initialization]]
Next: [[01_Learning_Rate_And_Schedulers]]
Related: [[10_Validation_Loop]], [[Training_Loop_Template]]
