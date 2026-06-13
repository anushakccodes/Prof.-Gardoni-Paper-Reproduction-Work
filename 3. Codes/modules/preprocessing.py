import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OUTPUT_NAMES = ['Dy', 'Vy', 'Du', 'Vu', 'IDy', 'IVy', 'IDu', 'IVu']

INPUT_COLS = [
    'B', 'D', 'H', 'B_num', 'D_num', 'H_num',
    'colWidth', 'beamDepth', 'beamRat', 'Asc', 'Asb',
    't', 'cover1', 'cover2', 'Ec', 'nu_c', 'fc',
    'fcuRat', 'eps_cu', 'Es', 'nu_s', 'fsy', 'eta'
]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(yaml_path):
    """Load hyperparameters.yaml and return parsed config components.

    Returns
    -------
    hp            : full dict
    DS            : dataset sub-dict
    DNN_DEFAULTS  : dnn_default sub-dict
    OUTS          : dnn_outputs sub-dict
    TRAIN_SPLIT   : float
    R2_THRESHOLD  : float
    INCREMENT     : int
    MAX_ITER      : int
    """
    with open(yaml_path) as f:
        hp = yaml.safe_load(f)

    DS           = hp['dataset']
    DNN_DEFAULTS = hp['dnn_default']
    OUTS         = hp['dnn_outputs']

    TRAIN_SPLIT  = DS['train_split']
    R2_THRESHOLD = DS['r2_threshold']
    INCREMENT    = DS['increment_per_iter']
    MAX_ITER     = DS['max_iterations']

    return hp, DS, DNN_DEFAULTS, OUTS, TRAIN_SPLIT, R2_THRESHOLD, INCREMENT, MAX_ITER


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_and_split(data_path, input_cols, output_names, train_split):
    """Load CSV and split using the pre-assigned 'split' column.

    The cleaned dataset (DNN_dataset_cleaned.csv, 110,649 rows) is the
    dataset used to produce the paper's results. The 'split' column
    pre-assigns each row to 'train' or 'test'.

    Returns
    -------
    X_train_all : np.ndarray (n_train, n_features) float32, scaled
    X_test      : np.ndarray (n_test,  n_features) float32, scaled
    Y_train_all : np.ndarray (n_train, n_outputs)  float32, unscaled
    Y_test      : np.ndarray (n_test,  n_outputs)  float32, unscaled
    scaler      : fitted StandardScaler
    """
    df = pd.read_csv(data_path)
    print(f'Loaded {len(df):,} rows x {len(df.columns)} columns')

    if 'split' in df.columns:
        train_df = df[df['split'] == 'train'].reset_index(drop=True)
        test_df  = df[df['split'] == 'test'].reset_index(drop=True)
        print(f'Using split column  --  train: {len(train_df):,}  |  test: {len(test_df):,}')
    else:
        train_df, test_df = train_test_split(
            df, test_size=1.0 - train_split, random_state=42
        )
        train_df = train_df.reset_index(drop=True)
        test_df  = test_df.reset_index(drop=True)
        print(f'Random split  --  train: {len(train_df):,}  |  test: {len(test_df):,}')

    scaler = StandardScaler()
    scaler.fit(train_df[input_cols])

    X_train_all = scaler.transform(train_df[input_cols]).astype('float32')
    X_test      = scaler.transform(test_df[input_cols]).astype('float32')
    Y_train_all = train_df[output_names].values.astype('float32')
    Y_test      = test_df[output_names].values.astype('float32')

    return X_train_all, X_test, Y_train_all, Y_test, scaler


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class RCFDataset(Dataset):
    """Minimal Dataset wrapper for numpy arrays."""

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def make_loaders(X_tr, y_tr, X_te, y_te, batch_size, device):
    """Wrap numpy arrays in RCFDataset and return (train_loader, test_loader).

    Train loader: shuffle=True,  batch_size
    Test  loader: shuffle=False, batch_size * 4
    """
    pin = device.type == 'cuda'
    train_loader = DataLoader(
        RCFDataset(X_tr, y_tr), batch_size=batch_size,
        shuffle=True, pin_memory=pin, num_workers=0
    )
    test_loader = DataLoader(
        RCFDataset(X_te, y_te), batch_size=batch_size * 4,
        shuffle=False, pin_memory=pin, num_workers=0
    )
    return train_loader, test_loader
