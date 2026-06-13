import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import r2_score

from modules.model import build_model
from modules.preprocessing import make_loaders, INPUT_COLS


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


# ---------------------------------------------------------------------------
# Single-model training
# ---------------------------------------------------------------------------

def train_one_model(output_name, X_tr, Y_tr, X_te, Y_te,
                    output_names, outs_cfg, defaults_cfg, device):
    """Train a single DNN for one output column.

    Parameters
    ----------
    output_name  : target name, e.g. 'Dy'
    X_tr, Y_tr   : training features and all-output labels (numpy, float32)
    X_te, Y_te   : test features and all-output labels (numpy, float32)
    output_names : ordered list of all output names
    outs_cfg     : dnn_outputs dict from hyperparameters.yaml
    defaults_cfg : dnn_default dict from hyperparameters.yaml
    device       : torch.device

    Returns
    -------
    model                    : best-checkpoint DNNModel
    train_losses, test_losses : per-epoch MSE lists
    train_r2, test_r2        : final R^2 scores
    y_tr_true, y_te_true     : ground-truth 1-D numpy arrays
    pred_tr, pred_te         : prediction 1-D numpy arrays
    epoch_times              : per-epoch wall-clock times (seconds)
    """
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
                ep_pred_tr = model(torch.from_numpy(X_tr).to(device)).cpu().numpy()
                ep_pred_te = model(torch.from_numpy(X_te).to(device)).cpu().numpy()
            ep_tr_r2     = r2_score(y_tr_1d.squeeze(), ep_pred_tr)
            ep_te_r2     = r2_score(y_te_1d.squeeze(), ep_pred_te)
            cumulative_t = time.perf_counter() - train_start
            print(f'    epoch {epoch+1:4d} | '
                  f'train MSE={train_losses[-1]:.6f} | test MSE={test_loss:.6f} | '
                  f'train R2={ep_tr_r2:.4f} | test R2={ep_te_r2:.4f} | '
                  f'cumulative time={_fmt_time(cumulative_t)}')

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
        pred_tr = model(torch.from_numpy(X_tr).to(device)).cpu().numpy()
        pred_te = model(torch.from_numpy(X_te).to(device)).cpu().numpy()

    train_r2 = r2_score(y_tr_1d.squeeze(), pred_tr)
    test_r2  = r2_score(y_te_1d.squeeze(), pred_te)

    return (model, train_losses, test_losses,
            train_r2, test_r2,
            y_tr_1d.squeeze(), y_te_1d.squeeze(),
            pred_tr, pred_te,
            epoch_times)


# ---------------------------------------------------------------------------
# Iterative training loop
# ---------------------------------------------------------------------------

def run_iterative_training(output_name, X_train_all, Y_train_all, X_test, Y_test,
                           increment, max_iter, r2_threshold,
                           output_names, outs_cfg, defaults_cfg, device):
    """Grow training set iteratively until R2 threshold is met or data exhausted.

    Parameters
    ----------
    output_name   : target name, e.g. 'Dy'
    X_train_all   : full scaled training features (numpy, float32)
    Y_train_all   : full training labels (numpy, float32)
    X_test        : test features (numpy, float32)
    Y_test        : test labels (numpy, float32)
    increment     : number of extra samples added per iteration
    max_iter      : maximum number of iterations
    r2_threshold  : R2 value at which to stop early
    output_names  : ordered list of all output names
    outs_cfg      : dnn_outputs dict from hyperparameters.yaml
    defaults_cfg  : dnn_default dict from hyperparameters.yaml
    device        : torch.device

    Returns
    -------
    final_result : dict with keys:
        model, train_losses, test_losses, train_r2, test_r2,
        y_tr_true, y_te_true, pred_tr, pred_te, r2_curve, final_n, epoch_times
    """
    SEP = "=" * 60
    print("\n" + SEP)
    print(f"  Training: {output_name}")
    print(SEP)

    r2_curve     = []
    final_result = None
    best_te_r2   = -float('inf')

    for iteration in range(1, max_iter + 1):
        n_samples = min(iteration * increment, len(X_train_all))
        X_tr = X_train_all[:n_samples]
        Y_tr = Y_train_all[:n_samples]

        iter_start = time.perf_counter()
        print(f"\n  -- iter {iteration:3d} | n={n_samples:7,} --")

        (model, tr_loss, te_loss,
         tr_r2, te_r2,
         y_tr_true, y_te_true,
         pred_tr, pred_te,
         epoch_times) = train_one_model(
            output_name, X_tr, Y_tr, X_test, Y_test,
            output_names, outs_cfg, defaults_cfg, device
        )

        iter_elapsed = time.perf_counter() - iter_start
        epoch_total  = sum(epoch_times)

        r2_curve.append((n_samples, te_r2))
        print(f"  iter {iteration:3d} DONE | n={n_samples:7,} | "
              f"train R2={tr_r2:.4f} | test R2={te_r2:.4f} | "
              f"iter time={iter_elapsed:.1f}s  (epochs total={epoch_total:.1f}s)")

        if te_r2 > best_te_r2:
            best_te_r2   = te_r2
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
                epoch_times  = epoch_times,
            )
            print(f"  *** New best model saved (test R2={te_r2:.4f})")
        else:
            print(f"  --- No improvement (best so far: test R2={best_te_r2:.4f})")

        if te_r2 >= r2_threshold:
            print(f"  >>> R2 threshold {r2_threshold} reached at n={n_samples:,}. Stopping.")
            break

        if n_samples >= len(X_train_all):
            print("  >>> Full training set used. Stopping.")
            break

    tr2 = final_result["train_r2"]
    te2 = final_result["test_r2"]
    fn  = final_result["final_n"]
    print(f"\n  Best model -- n={fn:,} | train R2={tr2:.4f}  test R2={te2:.4f}")

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
    axes[0].set_xlabel('Dataset size')
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
