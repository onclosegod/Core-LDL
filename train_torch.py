"""Run the repeated cross-validation protocol used for the main paper results.

For every outer fold, the script draws a 20% validation split from training
data, applies artificial missingness only to the remaining training labels,
selects checkpoints by validation Cosine, and evaluates once on the untouched
complete test labels. Split, fold, mask, and minibatch seeds are recorded with
all six LDL metrics. Output metadata also guards fold-level resume behavior.
"""
import argparse
import copy
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path

# Required by deterministic CUDA matrix multiplications. This must be set
# before PyTorch creates a CUDA context.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.distributed as dist

from myloss_psd_projection import MyLDLWrapper


DEFAULT_METRICS = ["chebyshev", "clark", "canberra", "kl_divergence", "cosine", "intersection"]
EPS = np.finfo(float).eps
SCRIPT_DIR = Path(__file__).resolve().parent
SHARED_DATASET_DIR = SCRIPT_DIR.parent / "dataset"


def seed_everything(seed):
    """Seed every random source used by the CORE-LDL training pipeline."""
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


# =========================
# Metrics
# =========================

def _random_mask(D, rate, weighted):
    if rate <= 0.0 or rate >= 1.0:
        raise ValueError("Invalid rate, which should be in the range (0, 1).")
    # Uniform missingness uses independent Bernoulli draws. The weighted mechanism
    # removes an exact global count without replacement, with probability proportional to 1-D.
    if weighted:
        p = 1 - D
        p = p / np.sum(p)
        select = np.random.choice(D.size, size=int(D.size * rate), replace=False, p=p.reshape(-1))
        mask = np.zeros_like(D, dtype=bool)
        mask.flat[select] = True
    else:
        mask = np.random.rand(*D.shape) < rate
    return mask


def random_missing(D, rate=0.8, weighted=False, return_mask=True):
    mask = _random_mask(D, rate, weighted)
    missing = D.copy()
    missing[mask] = 0.0
    return (missing, mask) if return_mask else missing


def load_dataset(name, dataset_dir=SHARED_DATASET_DIR):
    dataset_path = Path(dataset_dir) / f"{name}.mat"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    data = sio.loadmat(dataset_path)
    return data["features"], data["labels"]


def _clip_dist(D, D_pred):
    return np.clip(D, EPS, 1.0), np.clip(D_pred, EPS, 1.0)


def chebyshev(D, D_pred):
    return np.average(np.max(np.abs(D - D_pred), axis=1))


def clark(D, D_pred):
    D, D_pred = _clip_dist(D, D_pred)
    return np.average(np.sqrt(np.sum(((D - D_pred) ** 2) / ((D + D_pred) ** 2), axis=1)))


def canberra(D, D_pred):
    D, D_pred = _clip_dist(D, D_pred)
    return np.average(np.sum(np.abs(D - D_pred) / (D + D_pred), axis=1))


def kl_divergence(D, D_pred):
    D, D_pred = _clip_dist(D, D_pred)
    return np.average(np.sum(D * (np.log(D) - np.log(D_pred)), axis=1))


def cosine(D, D_pred):
    num = np.sum(D * D_pred, axis=1)
    den = np.sqrt(np.sum(D * D, axis=1)) * np.sqrt(np.sum(D_pred * D_pred, axis=1))
    return np.average(num / np.clip(den, EPS, None))


def intersection(D, D_pred):
    return np.average(np.sum(np.minimum(D, D_pred), axis=1))


def score(target, pred, metrics=None, return_dict=False):
    # These six LDL metrics are shared by training, ablations, and final reporting.
    metrics = metrics or DEFAULT_METRICS
    metric_map = {
        "chebyshev": chebyshev,
        "clark": clark,
        "canberra": canberra,
        "kl_divergence": kl_divergence,
        "cosine": cosine,
        "intersection": intersection,
    }
    scores = tuple(metric_map[m](target, pred) for m in metrics)
    return dict(zip(metrics, scores)) if return_dict else scores


# =========================
# Argument parsing
# =========================

def build_arg_parser(description="MyLDLWrapper K-fold cross-validation on LDL datasets"):
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument("--dataset", default="emotion6",
                        help="Dataset name (without .mat extension)")
    parser.add_argument("--dataset-dir", default=str(SHARED_DATASET_DIR))

    # Model
    parser.add_argument("--model-type", default="linear", choices=["linear", "mlp"])
    parser.add_argument("--activation", default="sigmoid", choices=["sigmoid", "tanh"],
                        help="Activation for relation head (correlation matrix R)")
    parser.add_argument("--covariance-mode", default="psd", choices=["psd", "diagonal"],
                        help="Covariance used by NLL/reconstruction; diagonal is for covariance ablation.")
    parser.add_argument("--num-bins", type=int, default=1000)
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension (mlp only)")

    # Training
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="Learning rate (auto: 2e-3 for linear, 1e-3 for mlp)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Batch size (auto: 512 for <=9 labels, 128 for >9)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early-stopping", type=int, default=0,
                        help="Early stopping patience (None to disable)")
    parser.add_argument("--use-scheduler", action="store_true")
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=1e-1)

    # Loss weights
    parser.add_argument("--w-kl", type=float, default=1.0)
    parser.add_argument("--w-l1", type=float, default=1.0)
    parser.add_argument("--w-impute", type=float, default=1.0)
    parser.add_argument("--w-loglik", type=float, default=1.0)

    # Missing rates
    parser.add_argument("--missing-rates", nargs="+", default=["0.1"],
                        help="Missing ratio(s), e.g. --missing-rates 0.1 or --missing-rates 0.1 0.2 0.3")
    parser.add_argument("--weighted-missing", action="store_true")

    # CV settings
    parser.add_argument("--n-outer-repeats", type=int, default=5)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)

    # Output
    parser.add_argument("--output-dir", default=None,
                        help="Override results directory")
    parser.add_argument("--no-resume", action="store_true",
                        help="Do not resume from previous results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print D_test/D_pred per fold")

    # GPU
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU device ID (single-GPU mode only)")
    parser.add_argument("--multi-gpu", action="store_true",
                        help="Use DistributedDataParallel on all GPUs (launch with: torchrun --nproc_per_node=N train_torch.py --multi-gpu ...)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set by torchrun)")

    return parser


def parse_args(argv=None, parser=None):
    parser = parser or build_arg_parser()
    args = parser.parse_args(argv)
    try:
        args.missing_rates = parse_missing_rates(args.missing_rates)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def parse_missing_rates(values):
    rates = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if not part:
                continue
            try:
                rate = float(part)
            except ValueError as exc:
                raise ValueError(f"invalid missing rate: {part}") from exc
            if rate < 0.0 or rate >= 1.0:
                raise ValueError("missing rates must be in [0, 1)")
            rates.append(rate)
    if not rates:
        raise ValueError("at least one missing rate is required")
    return rates


def protocol_metadata(args, loss_weights=None):
    """Return every setting that can change training or fold statistics."""
    weights = loss_weights or {
        "w_kl": args.w_kl,
        "w_l1": args.w_l1,
        "w_impute": args.w_impute,
        "w_loglik": args.w_loglik,
    }
    return {
        "model": args.model_type,
        "activation": args.activation,
        "covariance_mode": args.covariance_mode,
        "hidden_dim": int(args.hidden_dim),
        "num_bins": int(args.num_bins),
        "lr": float(args.lr),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "early_stopping": int(args.early_stopping),
        "use_scheduler": bool(args.use_scheduler),
        "scheduler_patience": int(args.scheduler_patience),
        "weight_decay": float(args.weight_decay),
        "w_kl": float(weights["w_kl"]),
        "w_l1": float(weights["w_l1"]),
        "w_impute": float(weights["w_impute"]),
        "w_loglik": float(weights["w_loglik"]),
        "weighted_missing": bool(args.weighted_missing),
        "n_outer_repeats": int(args.n_outer_repeats),
        "n_splits": int(args.n_splits),
        "random_seed": int(args.random_seed),
        "relation_alpha": float(getattr(args, "relation_alpha", 0.5)),
        "multi_gpu": bool(args.multi_gpu),
        "world_size": int(os.environ.get("WORLD_SIZE", "1")) if args.multi_gpu else 1,
    }


def protocol_fingerprint(args, loss_weights=None):
    payload = json.dumps(
        protocol_metadata(args, loss_weights),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]


def build_output_paths(args, missing_rate):
    """Build a stable output layout rooted at --output-dir."""
    if args.output_dir is None:
        if missing_rate > 0:
            output_root = SCRIPT_DIR / "outputs" / "main"
        else:
            output_root = SCRIPT_DIR / "outputs" / "main"
    else:
        output_root = Path(args.output_dir).expanduser()

    missing_rate_name = f"{missing_rate:.1f}"
    missing_dir = output_root / missing_rate_name

    cov_suffix = "" if args.covariance_mode == "psd" else f"_{args.covariance_mode}"
    if args.weighted_missing:
        run_name = f"torch_{args.dataset}_{args.model_type}_weighted_missing{cov_suffix}"
    else:
        run_name = f"torch_{args.dataset}_{args.model_type}_{args.activation}_com{cov_suffix}"
    run_name = f"{run_name}_{protocol_fingerprint(args)}"

    model_dir = missing_dir / run_name
    cv_scores_path = missing_dir / f"cv_scores_{run_name}.jsonl"

    missing_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    return output_root, missing_dir, model_dir, run_name, cv_scores_path


def build_fold_paths(model_dir, rep, fold_idx):
    prefix = f"rep{rep:02d}_fold{fold_idx:02d}"
    return {
        "full_model": model_dir / f"{prefix}_full_model.pt",
        "weights": model_dir / f"{prefix}_weights.pt",
        "metrics": model_dir / f"{prefix}_metrics.json",
        "history": model_dir / f"{prefix}_history.csv",
    }


def unwrap_torch_model(wrapper):
    return wrapper.model.module if getattr(wrapper, "is_data_parallel", False) else wrapper.model


def cpu_state_dict(module):
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in module.state_dict().items()
    }


def save_fold_artifacts(wrapper, record, fold_paths):
    base_model = unwrap_torch_model(wrapper)
    state_dict = cpu_state_dict(base_model)

    full_model = copy.deepcopy(base_model).to("cpu")
    torch.save(full_model, fold_paths["full_model"])
    torch.save(state_dict, fold_paths["weights"])

    record["artifacts"] = {name: str(path) for name, path in fold_paths.items()}
    with open(fold_paths["metrics"], "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def record_matches_run(record, args, missing_rate):
    """Accept cached folds only when protocol-defining metadata matches."""
    try:
        rec_missing_rate = float(record.get("missing_rate", missing_rate))
    except (TypeError, ValueError):
        return False

    return (
        record.get("dataset") == args.dataset
        and abs(rec_missing_rate - float(missing_rate)) < 1e-12
        and all(record.get(key) == value for key, value in protocol_metadata(args).items())
    )


def read_completed_folds(cv_scores_path, args, missing_rate):
    completed = set()
    if not Path(cv_scores_path).exists():
        return completed

    with open(cv_scores_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not record_matches_run(record, args, missing_rate):
                continue
            rep_done = record.get("rep")
            fold_done = record.get("fold")
            if isinstance(rep_done, int) and isinstance(fold_done, int):
                completed.add((rep_done, fold_done))
    return completed


def collect_metrics_by_fold(cv_scores_path, args, missing_rate):
    metrics_by_fold = {}
    if not Path(cv_scores_path).exists():
        return metrics_by_fold

    with open(cv_scores_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not record_matches_run(record, args, missing_rate):
                continue
            rep = record.get("rep")
            fold = record.get("fold")
            metrics = record.get("metrics", {})
            if not isinstance(rep, int) or not isinstance(fold, int):
                continue
            if not all(metric in metrics for metric in DEFAULT_METRICS):
                continue
            metrics_by_fold[(rep, fold)] = tuple(float(metrics[metric]) for metric in DEFAULT_METRICS)
    return metrics_by_fold


def write_summary_csv(metrics_by_fold, summary_path, args, missing_rate, run_name, expected_total):
    rows = []
    final_results = list(metrics_by_fold.values())
    for idx, metric_name in enumerate(DEFAULT_METRICS):
        values = [result[idx] for result in final_results]
        rows.append({
            "dataset": args.dataset,
            "run_name": run_name,
            "model": args.model_type,
            "activation": args.activation,
            "covariance_mode": args.covariance_mode,
            "relation_alpha": float(getattr(args, "relation_alpha", 0.5)),
            "missing_rate": float(missing_rate),
            "weighted_missing": bool(args.weighted_missing),
            "metric": metric_name,
            "mean": float(np.mean(values)) if values else "",
            "std": float(np.std(values)) if values else "",
            "n_folds": len(values),
            "expected_folds": expected_total,
        })

    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "run_name",
                "model",
                "activation",
                "covariance_mode",
                "relation_alpha",
                "missing_rate",
                "weighted_missing",
                "metric",
                "mean",
                "std",
                "n_folds",
                "expected_folds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def run_one_missing_rate(args, X, D, num_labels, dynamic_masks, device, local_rank, world_size, is_main, missing_rate):
    """Run and persist one repeated-CV experiment at a fixed missing rate.

    Every fold resets random states before constructing its validation split,
    missing mask, model, and minibatch order. Only rank zero evaluates the test
    fold and writes artifacts; all DDP workers participate in optimization.
    """
    output_root, missing_dir, model_dir, run_name, cv_scores_path = build_output_paths(args, missing_rate)
    summary_path = missing_dir / f"summary_{run_name}.csv"

    completed = set()
    if not args.no_resume:
        completed = read_completed_folds(cv_scores_path, args, missing_rate)

    mode_str = f"missing_rate={missing_rate:.1f}" if missing_rate > 0 else "no missing"
    if is_main:
        print(f"=== MyLDLWrapper({args.model_type}) CV | {mode_str} ===")
        print(f"Dataset: {args.dataset} | samples={X.shape[0]}, features={X.shape[1]}, labels={num_labels}")
        print(f"Params: lr={args.lr:.0e}, bins={args.num_bins}, epochs={args.epochs}, batch={args.batch_size}")
        print(f"Loss weights: kl={args.w_kl}, l1={args.w_l1}, impute={args.w_impute}, loglik={args.w_loglik}")
        print(f"Regularization: weight_decay={args.weight_decay}, early_stop={args.early_stopping}")
        print(f"Activation: {args.activation}")
        print(f"Relation blend alpha: {float(getattr(args, 'relation_alpha', 0.5)):g}")
        print(f"Covariance mode: {args.covariance_mode}")
        print(f"Mask: dynamic={dynamic_masks}, {mode_str}")
        print(f"CV: {args.n_outer_repeats}x{args.n_splits} folds, seed={args.random_seed}")
        print(f"Completed: {len(completed)}/{args.n_outer_repeats * args.n_splits}")
        if args.multi_gpu:
            print(f"GPU: DDP mode, world_size={world_size}, local_rank={local_rank}")
        else:
            print(f"GPU: device={device}")
        print(f"Output root: {output_root}")
        print(f"Missing-rate dir: {missing_dir}")
        print(f"Run name: {run_name}")
        print(f"Model artifacts: {model_dir}")
        print(f"CV scores: {cv_scores_path}")
        print(f"Summary CSV: {summary_path}")
        print()

    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    early_stopping_patience = args.early_stopping if args.early_stopping > 0 else None

    # The paper's default 5 repeats x 5 folds produces 25 held-out fold scores.
    for rep in range(args.n_outer_repeats):
        if is_main:
            print(f"===== missing_rate={missing_rate:.1f} | Repeat {rep + 1}/{args.n_outer_repeats}: {args.n_splits}-fold CV =====")

        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_seed + rep)

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(indices), start=1):
            if (rep + 1, fold_idx) in completed:
                if is_main:
                    print(f"[skip] missing_rate={missing_rate:.1f} rep {rep + 1} fold {fold_idx} already exists")
                continue

            # Reset all random states per fold so resumed runs reproduce the
            # same mask, initialization, minibatch order, and training path.
            split_seed = args.random_seed + rep * 100 + fold_idx
            fold_seed = args.random_seed + rep * 1000 + fold_idx
            seed_everything(fold_seed)

            fold_paths = build_fold_paths(model_dir, rep + 1, fold_idx)
            X_train_full, X_test = X[train_idx], X[test_idx]
            D_train_full, D_test = D[train_idx], D[test_idx]

            # Validation data is drawn only from the outer training fold; the test fold
            # remains untouched until the selected checkpoint is evaluated.
            X_train, X_val, D_train, D_val = train_test_split(
                X_train_full, D_train_full,
                test_size=0.2,
                random_state=split_seed,
            )

            mask_type = "dynamic mask" if dynamic_masks else "full mask"
            if is_main:
                print(f"[missing_rate={missing_rate:.1f} rep {rep + 1} fold {fold_idx}] training with {mask_type}")

            model = MyLDLWrapper(
                input_dim=X_train.shape[1],
                num_labels=D_train.shape[1],
                hidden_dim=args.hidden_dim,
                num_bins=args.num_bins,
                lr=args.lr,
                batch_size=args.batch_size,
                epochs=args.epochs,
                model_type=args.model_type,
                dynamic_masks=dynamic_masks,
                masks_per_sample=2,
                use_scheduler=args.use_scheduler,
                scheduler_factor=0.5,
                scheduler_patience=args.scheduler_patience,
                scheduler_min_lr=1e-6,
                w_kl=args.w_kl,
                w_l1=args.w_l1,
                w_impute=args.w_impute,
                w_loglik=args.w_loglik,
                weight_decay=args.weight_decay,
                use_multi_gpu=args.multi_gpu,
                gpu_ids=None if args.multi_gpu else [args.gpu],
                local_rank=local_rank,
                world_size=world_size,
                activation=args.activation,
                covariance_mode=args.covariance_mode,
                random_seed=fold_seed,
                relation_alpha=float(getattr(args, "relation_alpha", 0.5)),
            )

            # Apply experimental missingness only to training labels. Validation and test
            # labels remain complete for model selection and unbiased metric computation.
            mask = np.zeros_like(D_train, dtype=bool)
            if missing_rate > 0:
                _, mask = random_missing(D_train, rate=missing_rate, weighted=args.weighted_missing, return_mask=True)
            mask_val = np.zeros_like(D_val, dtype=bool)

            model.fit(
                X_train, D_train,
                mask=mask,
                X_val=X_val, D_val=D_val,
                mask_val=mask_val,
                grad_clip_norm=5.0,
                log_path=fold_paths["history"] if is_main else None,
                best_model_path=fold_paths["weights"] if is_main else None,
                best_metric="cosine",
                early_stopping_patience=early_stopping_patience,
                restore_best=True,
                verbose=is_main,
            )

            if is_main:
                D_pred = model.predict(X_test)

                if args.verbose:
                    print("D_test[:5]", D_test[:5])
                    print("D_pred[:5]", D_pred[:5])

                current_score = score(D_test, D_pred)

                # Store all seeds and protocol flags needed to audit or resume a fold.
                record = {
                    "framework": "torch",
                    "run_name": run_name,
                    "dataset": args.dataset,
                    "model": args.model_type,
                    "activation": args.activation,
                    "covariance_mode": args.covariance_mode,
                    "relation_alpha": float(getattr(args, "relation_alpha", 0.5)),
                    "output_root": str(output_root),
                    "missing_dir": str(missing_dir),
                    "artifact_dir": str(model_dir),
                    "missing_rate": float(missing_rate),
                    "weighted_missing": bool(args.weighted_missing),
                    "w_kl": float(args.w_kl),
                    "w_l1": float(args.w_l1),
                    "w_impute": float(args.w_impute),
                    "w_loglik": float(args.w_loglik),
                    "rep": int(rep + 1),
                    "fold": int(fold_idx),
                    "base_seed": int(args.random_seed),
                    "split_seed": int(split_seed),
                    "fold_seed": int(fold_seed),
                    "metrics": {metric: float(current_score[j]) for j, metric in enumerate(DEFAULT_METRICS)},
                }
                record.update(protocol_metadata(args))
                save_fold_artifacts(model, record, fold_paths)
                with open(cv_scores_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"[missing_rate={missing_rate:.1f} rep {rep + 1} fold {fold_idx}] Cosine = {current_score[4]:.4f}")

            if world_size > 1:
                dist.barrier()

    if is_main:
        print("\n" + "=" * 50)
        print(f"Dataset: {args.dataset} | Model: MyLDLWrapper({args.model_type}) | missing_rate={missing_rate:.1f}")
        print("=" * 50)

        metrics_by_fold = collect_metrics_by_fold(cv_scores_path, args, missing_rate)
        expected_total = args.n_outer_repeats * args.n_splits
        final_results = list(metrics_by_fold.values())
        print(f"Folds used for statistics: {len(final_results)}/{expected_total}")

        if not final_results:
            print("No metrics found for statistics. Please check the result path and config.")
        else:
            for idx, metric_name in enumerate(DEFAULT_METRICS):
                values = [result[idx] for result in final_results]
                print(f"{metric_name:12}: {np.mean(values):.4f}({np.std(values):.4f})")

        write_summary_csv(metrics_by_fold, summary_path, args, missing_rate, run_name, expected_total)
        print(f"Summary CSV saved to: {summary_path}")
        print()

    if world_size > 1:
        dist.barrier()


# =========================
# Main
# =========================

def main():
    args = parse_args()

    # ---- DDP initialization ----
    if args.multi_gpu:
        if 'RANK' not in os.environ and args.local_rank < 0:
            print("Error: --multi-gpu requires torchrun launch. Use: torchrun --nproc_per_node=N train_torch.py --multi-gpu ...")
            sys.exit(1)
        if args.local_rank >= 0:
            local_rank = args.local_rank
        else:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        dist.init_process_group(backend='nccl')
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        device = f'cuda:{local_rank}'
    else:
        local_rank = 0
        world_size = 1
        device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'

    is_main = (local_rank == 0)

    # Auto-detect lr
    if args.lr is None:
        args.lr = 2e-3 if args.model_type == "linear" else 1e-3

    # Auto-detect batch size based on label count
    # We need to load dataset first to know label count
    X, D = load_dataset(args.dataset, args.dataset_dir)
    num_labels = D.shape[1]

    # Enumerate the full artificial-mask pool for small label spaces; sample masks
    # dynamically when exhaustive enumeration would grow exponentially.
    dynamic_masks = num_labels > 9

    for missing_rate in args.missing_rates:
        run_one_missing_rate(
            args=args,
            X=X,
            D=D,
            num_labels=num_labels,
            dynamic_masks=dynamic_masks,
            device=device,
            local_rank=local_rank,
            world_size=world_size,
            is_main=is_main,
            missing_rate=missing_rate,
        )

    if world_size > 1:
        dist.destroy_process_group()
    return


if __name__ == "__main__":
    main()
