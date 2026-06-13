import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# DNN Architecture
# ---------------------------------------------------------------------------

class DNNModel(nn.Module):
    """Fully-connected DNN with optional BatchNorm + Dropout after each hidden layer.

    Parameters
    ----------
    n_inputs       : number of input features
    neurons        : list of hidden-layer widths, e.g. [128, 128, 64, 64, 64, 64]
    dropout_rate   : dropout probability applied after each hidden layer
    use_batch_norm : whether to insert BatchNorm1d after each Linear layer
    """

    def __init__(self, n_inputs, neurons, dropout_rate, use_batch_norm):
        super().__init__()
        layers = []
        in_dim = n_inputs
        for width in neurons:
            layers.append(nn.Linear(in_dim, width))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(width))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_dim = width
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(output_name, input_cols, outs_cfg, defaults_cfg, device):
    """Instantiate and return a DNNModel moved to *device*.

    Parameters
    ----------
    output_name  : one of the 8 target names, e.g. 'Dy'
    input_cols   : list of input feature column names (determines n_inputs)
    outs_cfg     : dnn_outputs dict from hyperparameters.yaml
    defaults_cfg : dnn_default dict from hyperparameters.yaml
    device       : torch.device
    """
    cfg = outs_cfg[output_name]
    model = DNNModel(
        n_inputs=len(input_cols),
        neurons=cfg['neurons_per_layer'],
        dropout_rate=cfg['dropout_rate'],
        use_batch_norm=defaults_cfg['batch_norm'],
    )
    return model.to(device)
