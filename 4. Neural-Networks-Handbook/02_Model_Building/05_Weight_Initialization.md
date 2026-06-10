# Weight Initialization

## The idea

Initialization sets the starting parameter values before training. PyTorch layers have reasonable defaults, but custom initialization can help when networks become deeper or training is unstable.

PyTorch provides initialization functions in `torch.nn.init`. ([PyTorch docs](https://docs.pytorch.org/docs/stable/nn.init.html))

## Why it matters

Bad initialization can make activations or gradients shrink, explode, or become uninformative. ReLU networks often use Kaiming initialization; tanh-style networks often use Xavier initialization.

## Mental model

Initialization is the model's starting point. It does not solve the problem, but a bad starting point can make the optimizer's job much harder.

## PyTorch example

```python
from torch import nn

layer = nn.Linear(10, 32)
nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
nn.init.zeros_(layer.bias)
```

## Research-style example

```python
def init_mlp_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

model.apply(init_mlp_weights)
```

## Common mistakes

- [ ] Re-initializing weights after loading a checkpoint.
- [ ] Using Kaiming initialization for a network with no ReLU-style activations.
- [ ] Initializing every parameter manually without checking PyTorch defaults first.
- [ ] Forgetting bias terms can be `None`.

## Previous / Next

Previous: [[01_Learning_Rate_And_Schedulers]]
Next: [[02_Overfitting_Underfitting]]
Related: [[01_Activation_Functions]], [[07_Debugging_Training]]

