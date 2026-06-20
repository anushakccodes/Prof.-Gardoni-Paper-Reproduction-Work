import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import redirect_stdout
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from modules.model import build_model
from modules.preprocessing import make_loaders, INPUT_COLS

# Fixed seed for reproducible weight initialisation and DataLoader shuffling.
# The paper does not state a seed; this value gives consistent run-to-run results.
GLOBAL_SEED = 42


class _Tee:
    """Write stdout to the notebook and a log file at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _fmt_time(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f'{seconds}s'
    elif seconds < 3600:
        return f'{seconds // 60}m {seconds % 60}s'
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        return f'{h}h {m}m {s}s'


def _print_epoch_log_header():
    print()
    print(
        f"    {'Epoch':>6}  "
        f"{'Train MSE':>12}  "
        f"{'Test MSE':>12}  "
        f"{'Train R2':>9}  "
        f"{'Test R2':>8}  "
        f"{'Time':>12}"
    )
    print(
        f"    {'-' * 6}  "
        f"{'-' * 12}  "
        f"{'-' * 12}  "
        f"{'-' * 9}  "
        f"{'-' * 8}  "
        f"{'-' * 12}"
    )


def _print_epoch_log_row(epoch, train_mse, test_mse, train_r2, test_r2, cumulative_time):
    print(
        f"    {epoch:6d}  "
        f"{train_mse:12.6f}  "
        f"{test_mse:12.6f}  "
        f"{train_r2:9.4f}  "
        f"{test_r2:8.4f}  "
        f"{_fmt_time(cumulative_time):>12}"
    )


def _save_best_checkpoint(model, output_name, iteration, n_samples, train_n, test_n,
                          train_r2, test_r2, input_scaler, target_scaler,
                          output_names, checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    path = checkpoint_dir / f"dnn_{output_name}_best.pt"

    torch.save(
        {
            "output_name": output_name,
            "iteration": iteration,
            "total_n": n_samples,
            "train_n": train_n,
            "test_n": test_n,
            "train_r2": train_r2,
            "test_r2": test_r2,
            "input_cols": INPUT_COLS,
            "output_names": output_names,
            "model_state_dict": {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            },
            "input_scaler_type": "MinMaxScaler",
            "input_scaler_min": input_scaler.min_,
            "input_scaler_scale": input_scaler.scale_,
            "input_scaler_data_min": input_scaler.data_min_,
            "input_scaler_data_max": input_scaler.data_max_,
            "input_scaler_data_range": input_scaler.data_range_,
            "target_scaler_type": "StandardScaler",
            "target_scaler_mean": target_scaler.mean_,
            "target_scaler_scale": target_scaler.scale_,
            "target_scaler_var": target_scaler.var_,
        },
        path,
    )

    return path


def _inverse_target(values, target_scaler):
    values = np.asarray(values).reshape(-1, 1)
    return target_scaler.inverse_transform(values).squeeze()


# ---------------------------------------------------------------------------
# Single-model training
# ---------------------------------------------------------------------------

def train_one_model(output_name, X_tr, Y_tr, X_te, Y_te,
                    output_names, outs_cfg, defaults_cfg, device,
                    target_scaler):
    """Train a single DNN for one output column.

    Parameters
    ----------
    output_name  : target name, e.g. 'Dy'
    X_tr, Y_tr   : normalized training features and labels (numpy, float32)
    X_te, Y_te   : normalized test features and labels (numpy, float32)
    output_names : ordered list of all output names
    outs_cfg     : dnn_outputs dict from hyperparameters.yaml
    defaults_cfg : dnn_default dict from hyperparameters.yaml
    device       : torch.device

    Returns
    -------
    model                    : best-checkpoint DNNModel
    train_losses, test_losses : per-epoch MSE lists
    train_r2, test_r2        : final R^2 scores
    y_tr_true, y_te_true     : ground-truth 1-D numpy arrays in original units
    pred_tr, pred_te         : prediction 1-D numpy arrays in original units
    epoch_times              : per-epoch wall-clock times (seconds)
    """
    torch.manual_seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    cfg          = outs_cfg[output_name]
    batch_size   = cfg['batch_size']
    weight_decay = cfg['weight_decay']
    max_epochs   = defaults_cfg['max_epochs']
    patience     = defaults_cfg['early_stopping_patience']
    lr           = defaults_cfg['initial_lr']

    out_idx  = output_names.index(output_name)
    y_tr_1d  = Y_tr[:, out_idx].reshape(-1, 1)
    y_te_1d  = Y_te[:, out_idx].reshape(-1, 1)

    train_loader, test_loader = make_loaders(
        X_tr, y_tr_1d, X_te, y_te_1d, batch_size, device
    )

    model     = build_model(output_name, INPUT_COLS, outs_cfg, defaults_cfg, device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses, test_losses = [], []
    epoch_times = []
    best_test    = float('inf')
    no_improve   = 0
    best_state   = None
    train_start  = time.perf_counter()
    _print_epoch_log_header()

    for epoch in range(max_epochs):
        epoch_start = time.perf_counter()

        # --- train pass ---
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).squeeze(-1)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(xb)
        train_losses.append(running / len(train_loader.dataset))

        # --- test pass (used for early stopping) ---
        model.eval()
        test_running = 0.0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                yb = yb.to(device).squeeze(-1)
                test_running += criterion(model(xb), yb).item() * len(xb)
        test_loss = test_running / len(test_loader.dataset)
        test_losses.append(test_loss)

        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_times.append(epoch_elapsed)

        # --- print every 500 epochs ---
        if (epoch + 1) % 500 == 0:
            with torch.no_grad():
                ep_pred_tr_scaled = model(torch.from_numpy(X_tr).to(device)).cpu().numpy()
                ep_pred_te_scaled = model(torch.from_numpy(X_te).to(device)).cpu().numpy()
            ep_tr_true = _inverse_target(y_tr_1d, target_scaler)
            ep_te_true = _inverse_target(y_te_1d, target_scaler)
            ep_pred_tr = _inverse_target(ep_pred_tr_scaled, target_scaler)
            ep_pred_te = _inverse_target(ep_pred_te_scaled, target_scaler)
            ep_tr_r2   = r2_score(ep_tr_true, ep_pred_tr)
            ep_te_r2   = r2_score(ep_te_true, ep_pred_te)
            cumulative_t = time.perf_counter() - train_start
            _print_epoch_log_row(
                epoch + 1,
                train_losses[-1],
                test_loss,
                ep_tr_r2,
                ep_te_r2,
                cumulative_t,
            )

        # --- early stopping ---
        if test_loss < best_test:
            best_test  = test_loss
            no_improve = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'    Early stopping at epoch {epoch+1}.')
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        pred_tr_scaled = model(torch.from_numpy(X_tr).to(device)).cpu().numpy()
        pred_te_scaled = model(torch.from_numpy(X_te).to(device)).cpu().numpy()

    y_tr_true = _inverse_target(y_tr_1d, target_scaler)
    y_te_true = _inverse_target(y_te_1d, target_scaler)
    pred_tr = _inverse_target(pred_tr_scaled, target_scaler)
    pred_te = _inverse_target(pred_te_scaled, target_scaler)

    train_r2 = r2_score(y_tr_true, pred_tr)
    test_r2  = r2_score(y_te_true, pred_te)

    return (model, train_losses, test_losses,
            train_r2, test_r2,
            y_tr_true, y_te_true,
            pred_tr, pred_te,
            epoch_times)


# ---------------------------------------------------------------------------
# Iterative training loop
# ---------------------------------------------------------------------------

def run_iterative_training(output_name, X_all, Y_all, train_split,
                           increment, max_iter, r2_threshold,
                           output_names, outs_cfg, defaults_cfg, device,
                           log_path=None, checkpoint_dir=None):
    """Run iterative training, optionally teeing stdout to log_path."""
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_fh:
            with redirect_stdout(_Tee(sys.stdout, log_fh)):
                print(f"Log file: {log_path}")
                return _run_iterative_training(
                    output_name, X_all, Y_all, train_split,
                    increment, max_iter, r2_threshold,
                    output_names, outs_cfg, defaults_cfg, device,
                    checkpoint_dir=checkpoint_dir,
                )

    return _run_iterative_training(
        output_name, X_all, Y_all, train_split,
        increment, max_iter, r2_threshold,
        output_names, outs_cfg, defaults_cfg, device,
        checkpoint_dir=checkpoint_dir,
    )


def _run_iterative_training(output_name, X_all, Y_all, train_split,
                            increment, max_iter, r2_threshold,
                            output_names, outs_cfg, defaults_cfg, device,
                            checkpoint_dir=None):
    """Grow total dataset size iteratively until R2 threshold is met or data exhausted.

    At each iteration, the first n rows are selected from the full dataset pool.
    That n-row subset is then split into train/test sets using train_split.
    A fresh MinMaxScaler is fitted on that iteration's training inputs only.
    A fresh StandardScaler is fitted on that iteration's training target only.
    If checkpoint_dir is provided, each new best model overwrites the stable
    best checkpoint file for this output.

    Parameters
    ----------
    output_name   : target name, e.g. 'Dy'
    X_all         : full unscaled feature pool (numpy, float32)
    Y_all         : full label pool (numpy, float32)
    train_split   : fraction of each iteration subset used for training
    increment     : number of extra samples added per iteration
    max_iter      : maximum number of iterations
    r2_threshold  : R2 value at which to stop early
    output_names  : ordered list of all output names
    outs_cfg      : dnn_outputs dict from hyperparameters.yaml
    defaults_cfg  : dnn_default dict from hyperparameters.yaml
    device        : torch.device
    checkpoint_dir: directory for new-best checkpoint files, or None

    Returns
    -------
    final_result : dict with keys:
        model, train_losses, test_losses, train_r2, test_r2,
        y_tr_true, y_te_true, pred_tr, pred_te, r2_curve, final_n,
        train_n, test_n, input_scaler, target_scaler, checkpoint_path, epoch_times
    """
    SEP = "=" * 60
    print("\n" + SEP)
    print(f"  TRAINING FOR OUTPUT: {output_name}")
    print(SEP)

    r2_curve     = []
    final_result = None
    best_te_r2   = -float('inf')

    for iteration in range(1, max_iter + 1):
        n_samples = min(iteration * increment, len(X_all))
        X_subset = X_all[:n_samples]
        Y_subset = Y_all[:n_samples]

        X_tr_raw, X_te_raw, Y_tr, Y_te = train_test_split(
            X_subset,
            Y_subset,
            train_size=train_split,
            random_state=GLOBAL_SEED,
            shuffle=True,
        )

        input_scaler = MinMaxScaler()
        X_tr = input_scaler.fit_transform(X_tr_raw).astype('float32')
        X_te = input_scaler.transform(X_te_raw).astype('float32')

        out_idx = output_names.index(output_name)
        target_scaler = StandardScaler()
        Y_tr_scaled = Y_tr.copy()
        Y_te_scaled = Y_te.copy()
        Y_tr_scaled[:, out_idx] = target_scaler.fit_transform(
            Y_tr[:, out_idx].reshape(-1, 1)
        ).squeeze()
        Y_te_scaled[:, out_idx] = target_scaler.transform(
            Y_te[:, out_idx].reshape(-1, 1)
        ).squeeze()

        iter_start = time.perf_counter()
        print(
            f"\n  ITERATION {iteration:3d} | TOTAL SAMPLES={n_samples:7,} "
            f"| TRAIN={len(X_tr):7,} | TEST={len(X_te):7,}"
        )

        (model, tr_loss, te_loss,
         tr_r2, te_r2,
         y_tr_true, y_te_true,
         pred_tr, pred_te,
         epoch_times) = train_one_model(
            output_name, X_tr, Y_tr_scaled, X_te, Y_te_scaled,
            output_names, outs_cfg, defaults_cfg, device,
            target_scaler=target_scaler,
        )

        iter_elapsed = time.perf_counter() - iter_start
        epoch_total  = sum(epoch_times)

        r2_curve.append((n_samples, te_r2))
        print(f"\n  ITER {iteration:3d} DONE | n={n_samples:7,} | "
              f"train R2={tr_r2:.4f} | test R2={te_r2:.4f} | "
              f"iter time={_fmt_time(iter_elapsed)}  (epochs total={_fmt_time(epoch_total)})")

        if te_r2 > best_te_r2:
            best_te_r2   = te_r2
            checkpoint_path = None
            if checkpoint_dir is not None:
                checkpoint_path = _save_best_checkpoint(
                    model=model,
                    output_name=output_name,
                    iteration=iteration,
                    n_samples=n_samples,
                    train_n=len(X_tr),
                    test_n=len(X_te),
                    train_r2=tr_r2,
                    test_r2=te_r2,
                    input_scaler=input_scaler,
                    target_scaler=target_scaler,
                    output_names=output_names,
                    checkpoint_dir=checkpoint_dir,
                )

            final_result = dict(
                model        = model,
                train_losses = tr_loss,
                test_losses  = te_loss,
                train_r2     = tr_r2,
                test_r2      = te_r2,
                y_tr_true    = y_tr_true,
                y_te_true    = y_te_true,
                pred_tr      = pred_tr,
                pred_te      = pred_te,
                r2_curve     = r2_curve,
                final_n      = n_samples,
                train_n      = len(X_tr),
                test_n       = len(X_te),
                input_scaler = input_scaler,
                target_scaler = target_scaler,
                checkpoint_path = checkpoint_path,
                epoch_times  = epoch_times,
            )
            if checkpoint_path is not None:
                print(
                    f"\n  *** Best model checkpoint updated "
                    f"(test R2={te_r2:.4f}): {checkpoint_path}"
                )
            else:
                print(f"\n  *** New best model saved with test R2={te_r2:.4f}")
        else:
            print(f"\n  --- No improvement (best so far: test R2={best_te_r2:.4f})")

        if te_r2 >= r2_threshold:
            print(f"\n  >>> R2 threshold {r2_threshold} reached at n={n_samples:,}. Stopping.")
            break

        if n_samples >= len(X_all):
            print("\n  >>> Full dataset pool used. Stopping.")
            break

    tr2 = final_result["train_r2"]
    te2 = final_result["test_r2"]
    fn  = final_result["final_n"]
    tn  = final_result["train_n"]
    vn  = final_result["test_n"]
    print(
        f"\n  Best model -- total n={fn:,} "
        f"(train={tn:,}, test={vn:,}) | train R2={tr2:.4f}  test R2={te2:.4f}"
    )

    return final_result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(output_name, result, r2_threshold, save_path=None):
    """1x4 subplot figure for a single output target.

    Columns:
      0 - R2 vs dataset size (convergence curve + threshold line)
      1 - Loss vs epoch (train & val, log scale)
      2 - Pred vs Actual (train scatter)
      3 - Pred vs Actual (test scatter)

    Parameters
    ----------
    output_name  : target name, e.g. 'Dy'
    result       : dict returned by run_iterative_training
    r2_threshold : threshold value drawn as a horizontal line in col 0
    save_path    : if not None, saves figure to this path
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 4.5))
    fig.suptitle(f'DNN Surrogate Model -- {output_name}',
                 fontsize=14, fontweight='bold')

    col_titles = [
        'R2 vs Dataset Size',
        'Loss vs Epoch',
        'Pred vs Actual (Train)',
        'Pred vs Actual (Test)',
    ]
    for col, title in enumerate(col_titles):
        axes[col].set_title(title, fontsize=11, fontweight='bold', pad=8)

    # -- Col 0: R2 convergence --
    sizes, r2s = zip(*result['r2_curve'])
    axes[0].plot(sizes, r2s, 'o-', color='steelblue', lw=1.5, ms=4)
    axes[0].axhline(r2_threshold, color='red', ls='--', lw=1,
                    label=f'threshold={r2_threshold}')
    axes[0].set_xlabel('Total subset size')
    axes[0].set_ylabel('Test R2')
    axes[0].set_ylim(max(0, min(r2s) - 0.05), 1.02)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(3, 3))

    # -- Col 1: Loss curve --
    epochs = range(1, len(result['train_losses']) + 1)
    axes[1].plot(epochs, result['train_losses'], label='Train', color='steelblue',  lw=1.2)
    axes[1].plot(epochs, result['test_losses'],  label='Test',  color='darkorange', lw=1.2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE Loss')
    axes[1].set_yscale('log')
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # -- Cols 2 & 3: Scatter plots --
    scatter_data = [
        (result['y_tr_true'], result['pred_tr'], result['train_r2']),
        (result['y_te_true'], result['pred_te'], result['test_r2']),
    ]
    for col_off, (true_vals, pred_vals, r2_val) in enumerate(scatter_data):
        ax = axes[2 + col_off]
        ax.scatter(true_vals, pred_vals, s=3, alpha=0.2,
                   color='steelblue', rasterized=True)
        lo = min(true_vals.min(), pred_vals.min())
        hi = max(true_vals.max(), pred_vals.max())
        ax.plot([lo, hi], [lo, hi], 'k-', lw=1.2, label='1:1')
        std = (pred_vals - true_vals).std()
        ax.plot([lo, hi], [lo + std, hi + std], 'r--', lw=0.8, label='+/-1 std')
        ax.plot([lo, hi], [lo - std, hi - std], 'r--', lw=0.8)
        ax.set_xlabel(f'True {output_name}')
        ax.set_ylabel(f'Predicted {output_name}')
        ax.text(0.05, 0.92, f'R2 = {r2_val:.4f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', fc='lightyellow', alpha=0.8))
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Figure saved: {save_path}')

    plt.show()


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model, output_name, save_dir):
    """Save model state_dict to save_dir/dnn_{output_name}.pt.

    Parameters
    ----------
    model       : trained DNNModel instance
    output_name : target name, e.g. 'Dy'
    save_dir    : pathlib.Path or str; directory will be created if absent
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    path = save_dir / f'dnn_{output_name}.pt'
    torch.save(model.state_dict(), path)
    print(f'Saved: {path}')
