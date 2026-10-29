"""CORE-LDL model, probabilistic objective, training wrapper, and inference.

Paper-to-code symbol map:
    ``output`` is the label-wise bin distribution G.
    ``point`` is the simplex-normalized marginal mean mu.
    ``sigma`` is the label-wise dispersion.
    ``R`` is first a vector of pairwise affinities and later the PSD correlation.
    ``Sigma`` is the joint covariance diag(sigma) R diag(sigma).
    ``loss1``--``loss4`` are L_KL, L_L1, L_rec, and L_NLL, respectively.

Stable LaTeX labels are used below instead of equation numbers, which can change
during typesetting.
"""
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import csv
import copy
import os
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from impute_loss import  batch_convert_to_correlation_matrix, find_duplicate_indices
from impute_loss import generate_random_masks_for_batch, generate_masks
from impute_loss import impute_loss_dynamic, impute_loss
from impute_loss import impute_loss_dynamic_batch, impute_loss_parallel
from impute_loss import graph_loss, label_correlation_consistency_loss
from torch.optim import lr_scheduler

def log_likehood_loss(Sigma, mu, target, jitter=1e-4):
    """Compute the marginalized Gaussian NLL on originally observed labels.

    Paper mapping: ``eq:observed_nll``, ``eq:identity_padding``, and
    ``app:batched_nll``. NaNs encode original missingness. Identity padding
    permits one fixed-shape batched solve and changes the objective only by a
    parameter-independent constant. The return value includes the fixed
    internal factor 0.1 reported with the overall objective.

    Args:
        Sigma: Joint covariance, shape ``(batch, labels, labels)``.
        mu: Simplex-normalized marginal mean, shape ``(batch, labels)``.
        target: Incomplete labels with NaNs at originally missing coordinates.
        jitter: Diagonal stabilization constant tau used by the NLL solver.

    Returns:
        Per-instance NLL values with shape ``(batch,)``.
    """
    batch_size, num_labels = target.shape
    device = target.device

    # NaN marks an originally unobserved label degree and must not contribute to the NLL.
    nan_mask = torch.isnan(target)          # (B, L)
    obs_mask = ~nan_mask                    # (B, L)
    obs_cross = obs_mask.unsqueeze(2) & obs_mask.unsqueeze(1)  # (B, L, L)

    # Retain the observed covariance block and fill padded dimensions with identity.
    eye = torch.eye(num_labels, device=device).unsqueeze(0)  # (1, L, L)
    # Identity padding makes the full matrix algebra equal to the observed-subset NLL
    # up to a parameter-independent constant, while retaining efficient batched solves.
    Sigma_padded = torch.where(obs_cross, Sigma, eye)        # (B, L, L)

    # This NLL-specific shift is distinct from reconstruction stabilization.
    jitter_eye = eye * float(jitter)
    Sigma_stable = Sigma_padded + jitter_eye                 # (B, L, L)

    # Solve the stabilized system directly for the covariance-weighted residual.
    residual = torch.where(nan_mask, torch.zeros_like(mu), target - mu)  # (B, L), 0 at missing dims
    r_t = residual.unsqueeze(-1)                              # (B, L, 1)
    Sigma_inv_r = torch.linalg.solve(Sigma_stable, r_t)      # (B, L, 1)
    quad = torch.bmm(r_t.transpose(1, 2), Sigma_inv_r).squeeze(-1).squeeze(-1)  # (B,)

    # Batched log-determinant (does not require inverse)
    log_det = torch.linalg.slogdet(Sigma_stable).logabsdet   # (B,)

    nll = (quad + log_det) / 2.0
    return nll / 10.0

def normal_sampling(mean, label_k, std=2):
    """Evaluate the Gaussian KL target at one support point.

    Paper mapping: ``eq:gaussian_target`` and ``app:kl_target``. The caller
    normalizes these values across bins to obtain G_tar.
    """
    std = torch.clamp(std, min=1e-4)  # Prevent a degenerate zero-width target.
    return torch.exp(-(label_k-mean)**2/(2*std**2)) / (math.sqrt(2*math.pi)*std)

def L1_loss_mask(inputs, labels, mask):
    """Compute observed-coordinate mean alignment (``eq:masked_l1``)."""
    batch_size = labels.shape[0]
    inputs = inputs.reshape(batch_size, -1)
    criterion = nn.L1Loss(reduction='none')
    loss = criterion(inputs, labels.float())
    loss = (loss * mask) / (mask.sum() + 1e-8)
    return loss.sum()

def kl_loss_mask(inputs, labels, weight):
    """Compute observed label-bin KL regularization (``eq:masked_kl``)."""
    batch_size = labels.shape[0]
    inputs, labels, weight = inputs.reshape(-1), labels.reshape(-1), weight.reshape(-1)
    criterion = nn.KLDivLoss(reduction="none")
    outputs = torch.log(inputs + 1e-8)  # Keep log probabilities finite.
    loss = criterion(outputs, labels)
    # Fixed internal scale used to balance KL with the other objectives.
    loss = 10 * (loss * weight).sum() / (weight.sum() + 1e-8)
    return loss

def L1_loss(inputs, labels):
    """Legacy unmasked L1 helper; reported runs use ``L1_loss_mask``."""
    batch_size = labels.shape[0]
    inputs = inputs.reshape(batch_size, -1)
    criterion = nn.L1Loss(reduction='mean')
    loss = criterion(inputs, labels.float())
    return loss

def project_to_psd_correlation(R, eps=1e-6):
    """Project affinity templates to PSD unit-diagonal correlation matrices.

    Paper mapping: ``eq:psd_relation`` and ``app:psd_stabilization``.
    Symmetrization, eigenvalue flooring, and diagonal renormalization remain
    differentiable with respect to the input affinity template.
    """
    R = 0.5 * (R + R.transpose(1, 2))
    eigvals, eigvecs = torch.linalg.eigh(R)
    eigvals = eigvals.clamp_min(float(eps))
    R_psd = torch.matmul(eigvecs * eigvals.unsqueeze(1), eigvecs.transpose(1, 2))
    diag = torch.diagonal(R_psd, dim1=1, dim2=2).clamp_min(float(eps)).sqrt()
    R_corr = R_psd / (diag.unsqueeze(2) * diag.unsqueeze(1)).clamp_min(float(eps))
    R_corr = 0.5 * (R_corr + R_corr.transpose(1, 2))
    idx = torch.arange(R_corr.shape[-1], device=R_corr.device)
    R_corr[:, idx, idx] = 1.0
    return R_corr

class myloss(nn.Module):
    """Construct the CORE-LDL covariance and four-term training objective.

    Paper mapping: main-paper labels ``eq:marginal_moments`` through
    ``eq:overall_objective``. Fixed artificial subsets are used for the reported
    datasets with at most nine labels; dynamic masks are a large-label fallback.
    """
    def __init__(self, rang=1, skip=0.05, num_bins=20, num_labels=14, dynamic_masks=True,
                    masks_per_sample=4, missing_ratio_range=(0.3, 0.7),
                    w_kl=1.0, w_l1=1.0, w_impute=1.0, w_loglik=1.0,
                    nll_jitter=1e-4,
                    psd_eps=1e-6, covariance_mode="psd",
                    use_em_fill=True, em_fill_warmup=0,
                    relation_alpha=0.5):
        super(myloss, self).__init__()
        if covariance_mode not in {"psd", "diagonal"}:
            raise ValueError("covariance_mode must be 'psd' or 'diagonal'")
        if not 0.0 <= float(relation_alpha) <= 1.0:
            raise ValueError("relation_alpha must be in [0, 1]")
        self.register_buffer('rank', torch.Tensor([i for i in np.arange(0, rang, skip)]).reshape(1,-1))
        self.skip = skip
        self.num_labels = num_labels
        self.dynamic_masks = dynamic_masks
        self.masks_per_sample = masks_per_sample
        self.missing_ratio_range = missing_ratio_range
        self.num_bins = num_bins
        self.w_kl = w_kl
        self.w_l1 = w_l1
        self.w_impute = w_impute
        self.w_loglik = w_loglik
        self.nll_jitter = float(nll_jitter)
        self.psd_eps = float(psd_eps)
        self.covariance_mode = covariance_mode
        self.use_em_fill = use_em_fill
        self.em_fill_warmup = em_fill_warmup
        self.relation_alpha = float(relation_alpha)
        self.current_epoch = 0

        if not dynamic_masks:
            self.register_buffer(
                'masks',
                generate_masks(
                    num_labels=num_labels
                ),
            )
        else:
            self.masks = None

        self.loss = nn.L1Loss(reduction='none')

    def forward(self, output, target, R, rang=1, modes='train_only', hidden=None):
        """Evaluate CORE-LDL losses for one mini-batch.

        Args:
            output: Label-bin probabilities G, shape ``(batch, bins, labels)``.
            target: Labels using ``-1`` at originally missing coordinates.
            R: Pairwise affinity vectors emitted by the relation head.
            modes: ``train_only`` returns all four losses; other values return
                reconstruction plus NLL for validation bookkeeping.

        Returns:
            A loss dictionary in training mode, otherwise a scalar validation
            objective containing reconstruction and observed-subset NLL.
        """
        device = output.device
        
        batch_size = output.shape[0]
        
        if self.dynamic_masks:
            masks_3d = generate_random_masks_for_batch(
                batch_size, 
                self.num_labels, 
                masks_per_sample=self.masks_per_sample,
                missing_ratio_range=self.missing_ratio_range
            )
            masks_3d = masks_3d.to(device)
        
        # Convert the wrapper's -1 sentinel to the NaN convention used by all masks.
        target = target.clone()
        neg1_mask = torch.eq(target, -1)
        target[neg1_mask] = float('nan')

        batch_size = output.shape[0]
        output = output.reshape(batch_size, rang*self.num_bins, -1)
        
        # Paper ``eq:marginal_moments``: obtain mu and sigma from label-wise bins.
        point = torch.matmul(self.rank, output).reshape(batch_size, -1)
        point = point / point.sum(dim=1, keepdim=True).clamp(min=1e-8)

        variance = torch.sum(((self.rank - point.reshape(batch_size,-1,1))**2) * output.permute(0,2,1), dim=2)
        sigma = torch.sqrt(torch.clamp(variance, min=1e-8)).reshape(batch_size, -1)
        
        num_labels = sigma.shape[1]
        ss = sigma   
        Sigma = torch.matmul(ss.reshape(batch_size, num_labels, 1), ss.reshape(batch_size, 1, num_labels))
        # Paper ``eq:relation_blending``: combine instance and batch affinities.
        R = (
            self.relation_alpha * R
            + (1.0 - self.relation_alpha) * R.mean(dim=0, keepdim=True)
        )
        # Paper ``eq:psd_relation``: expand the affinity vector, then project it.
        if self.covariance_mode == "psd":
            R = project_to_psd_correlation(batch_convert_to_correlation_matrix(R, num_labels), eps=self.psd_eps)
        else:
            R = torch.eye(num_labels, device=Sigma.device, dtype=Sigma.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        # Paper ``eq:joint_gaussian``: elementwise scaling implements
        # diag(sigma) @ R @ diag(sigma).
        Sigma = R * Sigma  
        if self.dynamic_masks:
            masks_to_use = masks_3d
        else:
            indict = [i for i in range(self.masks.shape[0])]
            random.shuffle(indict)
            masks_to_use = self.masks[indict, :]

        # Paper ``eq:filled_target``: observed degrees retain ground truth and
        # originally missing degrees receive a detached current mean. This dense
        # target is exclusive to L_rec and does not create direct supervision.
        if self.use_em_fill and self.current_epoch >= self.em_fill_warmup:
            target_for_impute = torch.where(torch.isnan(target), point.detach(), target)
        else:
            target_for_impute = target

        if modes == 'train_only':
            if self.dynamic_masks:
                loss3 = impute_loss_dynamic_batch(output, sigma, Sigma, point, target_for_impute, masks_to_use)
            else:
                loss3 = impute_loss_parallel(output, sigma, Sigma, point, target_for_impute, masks_to_use)
            # Paper ``eq:observed_nll``: score the original observed subset,
            # never the detached values inserted into the reconstruction target.
            loss4 = log_likehood_loss(Sigma, point, target, jitter=self.nll_jitter).mean()

            # Paper ``eq:gaussian_target``: construct Gaussian bin targets from
            # observed degrees. Detaching sigma prevents KL from widening its target.
            with torch.no_grad():
                nan_mask = torch.isnan(target)
                weight = torch.where(nan_mask, torch.tensor(0.0, device=device), torch.tensor(1.0, device=device))
                target_clean = torch.nan_to_num(target)  
                
                weights = [weight for i in np.arange(0, rang, self.skip)]
                weights = torch.stack(weights, dim=1)
                
                detached_sigma = sigma.detach() 
                label = [normal_sampling(target_clean, i, detached_sigma) for i in np.arange(0, rang, self.skip)]
                label = torch.stack(label, dim=1)
                label = F.normalize(label, p=1, dim=1)
                
            loss2_pre = L1_loss_mask(point, target_clean, weight)
            loss2 = loss2_pre
 
            loss1 = kl_loss_mask(output, label, weights)

        else:
            if self.dynamic_masks:
                loss3 = impute_loss_dynamic_batch(output, sigma, Sigma, point, target_for_impute, masks_to_use)
            else:
                loss3 = impute_loss_parallel(output, sigma, Sigma, point, target_for_impute, masks_to_use)
            loss4 = log_likehood_loss(Sigma, point, target, jitter=self.nll_jitter).mean()

        if modes != "train_only":
            return loss3 + loss4 
        else:
            # Paper ``eq:overall_objective``: combine KL, L1, reconstruction, and NLL.
            total_loss = (self.w_kl * loss1 + self.w_l1 * loss2
                          + self.w_impute * loss3 + self.w_loglik * loss4)
            return {
                'loss': total_loss,
                'loss_kl': loss1,
                'loss_l1': loss2,
                'loss_impute': loss3,
                'loss_loglikelihood': loss4,
            }


class LinearLDLNet(nn.Module):
    """Parallel marginal and relation heads used in the reported experiments.

    Paper mapping: ``eq:marginal_head`` and ``eq:local_relation``. Both affine
    projections act directly on the input feature vector. The marginal logits
    are reshaped to ``(batch, bins, labels)`` and normalized over bins; the
    relation branch emits the strict-upper-triangle affinity vector.
    """
    def __init__(self, input_dim, num_labels, num_bins=20, activation='sigmoid'):
        super().__init__()
        self.num_labels = num_labels
        self.num_bins = num_bins
        self.num_rel = num_labels * (num_labels - 1) // 2

        act_fn = nn.Sigmoid() if activation == 'sigmoid' else nn.Tanh()

        self.regressor = nn.Sequential(
            nn.Linear(input_dim, num_labels * num_bins),
            nn.Dropout(0.1)
        )
        self.relation = nn.Sequential(
            nn.Linear(input_dim, self.num_rel),
            act_fn,
        )

    def forward(self, x):
        """Return G and pairwise affinities for an input feature batch."""
        pred_dist = self.regressor(x).view(-1, self.num_bins, self.num_labels)
        pred_dist = torch.softmax(pred_dist, dim=1)
        R = self.relation(x)
        return pred_dist, R


class MLP_LDLNet(nn.Module):
    """Nonlinear-backbone variant retained for controlled comparisons.

    This variant is not the linear architecture used for the reported main
    results. It applies the same two heads to a shared learned representation.
    """
    def __init__(self, input_dim, num_labels, num_bins=20, hidden_dim=128, activation='sigmoid'):
        super().__init__()
        self.num_labels = num_labels
        self.num_bins = num_bins
        self.num_rel = num_labels * (num_labels - 1) // 2

        act_fn = nn.Sigmoid() if activation == 'sigmoid' else nn.Tanh()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, num_labels * num_bins),
            nn.Dropout(0.1)
        )
        self.relation = nn.Sequential(
            nn.Linear(hidden_dim, self.num_rel),
            act_fn,
        )

    def forward(self, x):
        """Return G, pairwise affinities, and the shared MLP representation."""
        feat = self.backbone(x)
        pred_dist = self.regressor(feat).view(-1, self.num_bins, self.num_labels)
        pred_dist = torch.softmax(pred_dist, dim=1)
        R = self.relation(feat)
        return pred_dist, R, feat

class MyLDLWrapper():
    """Train and evaluate CORE-LDL with optional distributed data parallelism.

    The wrapper converts external boolean missing masks to the internal ``-1``
    sentinel, owns optimization/checkpoint selection, and exposes the inference
    rule associated with ``eq:prediction``.
    """
    def __init__(self, input_dim, num_labels, hidden_dim=128, num_bins=20, lr=1e-3,
                 batch_size=128, epochs=30, device=None, model_type='mlp',
                 use_multi_gpu=True, masks_per_sample = 2, gpu_ids=None, dynamic_masks=False,
                 use_scheduler=True, scheduler_factor=0.5, scheduler_patience=3, scheduler_min_lr=1e-6,
                 w_kl=1.0, w_l1=1.0, w_impute=1.0, w_loglik=1.0,
                 nll_jitter=1e-4,
                 psd_eps=1e-6,
                  covariance_mode="psd",
                   weight_decay=1e-1, local_rank=0, world_size=1, activation='sigmoid',
                   use_em_fill=True, em_fill_warmup=0, random_seed=None,
                   relation_alpha=0.5):
        self.random_seed = int(torch.initial_seed() % (2 ** 32)) if random_seed is None else int(random_seed)
        if random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_seed)
        self.input_dim = input_dim
        self.num_labels = num_labels
        self.hidden_dim = hidden_dim
        self.num_bins = num_bins
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self.model_type = model_type
        self.use_multi_gpu = use_multi_gpu
        self.gpu_ids = gpu_ids
        self.local_rank = local_rank
        self.world_size = world_size
        self.nll_jitter = float(nll_jitter)
        self.psd_eps = float(psd_eps)
        self.covariance_mode = covariance_mode
        self.relation_alpha = float(relation_alpha)

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            if world_size > 1:
                self.device = f'cuda:{local_rank}'
                torch.cuda.set_device(local_rank)
            else:
                self.device = f'cuda:{gpu_ids[0] if gpu_ids else 0}'
        else:
            self.device = 'cpu'

        if self.model_type == 'linear':
            model = LinearLDLNet(
                input_dim=input_dim,
                num_labels=num_labels,
                num_bins=num_bins,
                activation=activation,
            )
        elif self.model_type == 'mlp':
            model = MLP_LDLNet(
                input_dim=input_dim,
                num_labels=num_labels,
                num_bins=num_bins,
                hidden_dim=hidden_dim,
                activation=activation,
            )
        else:
            raise ValueError("model_type must be 'linear' or 'mlp'")

        # R is fixed to the identity in the diagonal ablation, so the relation
        # head is intentionally inactive. Freezing it keeps the original
        # numerical behavior and prevents DDP from expecting unused gradients.
        if self.covariance_mode == "diagonal":
            for parameter in model.relation.parameters():
                parameter.requires_grad_(False)

        # Move model to device BEFORE DDP wrapping (DDP requires model on target device)
        model = model.to(self.device)

        # Optional multi-GPU via DistributedDataParallel
        self.is_data_parallel = False
        if self.world_size > 1 and self.device.startswith('cuda'):
            model = DDP(model, device_ids=[self.local_rank])
            self.is_data_parallel = True

        self.model = model
        # The criterion owns all four paper losses and the PSD covariance construction.
        self.criterion = myloss(
            rang=1,
            skip=1.0 / num_bins,
            num_labels= num_labels,
            num_bins= num_bins,
            masks_per_sample= masks_per_sample,
            dynamic_masks = dynamic_masks,
            w_kl=w_kl,
            w_l1=w_l1,
            w_impute=w_impute,
            w_loglik=w_loglik,
            nll_jitter=nll_jitter,
            psd_eps=self.psd_eps,
            covariance_mode=self.covariance_mode,
            use_em_fill=use_em_fill,
            em_fill_warmup=em_fill_warmup,
            relation_alpha=self.relation_alpha,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=weight_decay)
        self.use_scheduler = bool(use_scheduler)
        self.scheduler = None
        if self.use_scheduler:
            self.scheduler = lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=float(scheduler_factor),
                patience=int(scheduler_patience),
                min_lr=float(scheduler_min_lr),
            )
    @staticmethod
    def _cosine_np(D, D_pred, eps=1e-12):
        D = np.clip(D, eps, 1.0)
        D_pred = np.clip(D_pred, eps, 1.0)
        num = np.sum(D * D_pred, axis=1)
        den = np.sqrt(np.sum(D * D, axis=1)) * np.sqrt(np.sum(D_pred * D_pred, axis=1))
        return float(np.average(num / np.clip(den, eps, None)))

    def fit(
        self,
        X,
        D,
        mask=None,
        *,
        X_val=None,
        D_val=None,
        mask_val=None,
        grad_clip_norm=5.0,
        log_path=None,
        best_model_path=None,
        best_metric="cosine",
        early_stopping_patience=None,
        early_stopping_min_delta=1e-4,
        restore_best=True,
        verbose=True,
    ):
        """Optimize the four-term objective and select a validation checkpoint.

        ``mask=True`` denotes an originally missing training coordinate. The
        wrapper replaces these entries by ``-1`` before the criterion converts
        them to NaN. Validation Cosine is the reported default selection rule;
        the held-out test fold is never passed to this method.

        Args:
            X, D: Training features and complete label distributions before the
                experimental missing mask is applied.
            mask: Boolean training mask with the same shape as D.
            X_val, D_val: Optional validation split drawn from the outer training fold.
            mask_val: Optional validation missing mask; main runs use a full mask.
            grad_clip_norm: Maximum gradient norm, or ``None`` to disable clipping.
            log_path: Optional per-epoch CSV path.
            best_model_path: Optional path for the selected state dictionary.
            best_metric: ``cosine`` to maximize or ``loss`` to minimize.
            restore_best: Restore the selected validation checkpoint before return.
        """

        # Keep numpy copies for metrics/logging
        X_val_np = None if X_val is None else np.asarray(X_val)
        D_val_np = None if D_val is None else np.asarray(D_val)

        X = torch.tensor(X, dtype=torch.float32)
        D = torch.tensor(D, dtype=torch.float32)

        # True entries are removed from supervision and converted to the internal -1 marker.
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool)
            D = D.clone()
            D[mask] = -1.0

        if X_val is not None and D_val is not None:
            X_val_t = torch.tensor(X_val_np, dtype=torch.float32)
            D_val_t = torch.tensor(D_val_np, dtype=torch.float32)
            if mask_val is not None:
                mask_val = torch.tensor(mask_val, dtype=torch.bool)
                D_val_t = D_val_t.clone()
                D_val_t[mask_val] = -1.0
        else:
            X_val_t, D_val_t = None, None
        train_dataset = TensorDataset(X, D)
        train_sampler = None
        if self.world_size > 1:
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank,
                shuffle=True,
                seed=self.random_seed,
            )
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=train_sampler,
            )
        else:
            # A dedicated generator makes minibatch order reproducible for a fold seed.
            data_loader_generator = torch.Generator()
            data_loader_generator.manual_seed(self.random_seed)
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                generator=data_loader_generator,
            )

        val_loader = None
        if X_val_t is not None and D_val_t is not None and self.local_rank == 0:
            val_loader = DataLoader(
                TensorDataset(X_val_t, D_val_t),
                batch_size=self.batch_size,
                shuffle=False,
            )

        # Prepare csv logger
        csv_file = None
        csv_writer = None
        if log_path is not None:
            log_path = str(log_path)
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            csv_file = open(log_path, "w", newline="")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "epoch",
                    "train_loss",
                    "train_loss_kl",
                    "train_loss_l1",
                    "train_loss_impute",
                    "train_loss_loglikelihood",
                    # "train_loss_label_correlation",
                    # "train_loss_graph",
                    "val_loss",
                    "val_loss_kl",
                    "val_loss_l1",
                    "val_loss_impute",
                    "val_loss_loglikelihood",
                    # "val_loss_label_correlation",
                    # "val_loss_graph",
                    "val_cosine",
                    "best_metric",
                    "best_score",
                    "lr",
                ],
            )
            csv_writer.writeheader()

        history = []

        # Best model tracking
        best_state = None
        best_score = None
        if best_metric not in {"cosine", "loss"}:
            raise ValueError("best_metric must be 'cosine' or 'loss'")

        if val_loader is not None:
            if best_metric == "loss":
                best_score = float("inf")
            else:
                best_score = -float("inf")
        else:
            best_metric = "loss"
            best_score = float("inf")

        use_early_stopping = (
            val_loader is not None
            and early_stopping_patience is not None
            and int(early_stopping_patience) > 0
        )
        no_improve_epochs = 0
        stop_training = False

        self.model.train()
        for epoch in range(self.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # -------- train --------
            sums = {
                "loss": 0.0,
                "loss_kl": 0.0,
                "loss_l1": 0.0,
                "loss_impute": 0.0,
                "loss_loglikelihood": 0.0,
                # "loss_label_correlation":0.0,
                # "loss_graph":0.0
            }

            for xb, db in train_loader:
                xb = xb.to(self.device)
                db = db.to(self.device)
                if self.model_type == 'linear':
                    pred_dist, R = self.model(xb)
                    loss_dict = self.criterion(pred_dist, db, R, modes="train_only", hidden=xb)
                elif self.model_type == 'mlp':
                    pred_dist, R, hidden = self.model(xb)
                    loss_dict = self.criterion(pred_dist, db, R, modes="train_only", hidden=hidden)
                loss = loss_dict["loss"]

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=float(grad_clip_norm))
                self.optimizer.step()

                sums["loss"] += float(loss_dict["loss"].item())
                sums["loss_kl"] += float(loss_dict["loss_kl"].item())
                sums["loss_l1"] += float(loss_dict["loss_l1"].item())
                sums["loss_impute"] += float(loss_dict["loss_impute"].item())
                sums["loss_loglikelihood"] += float(loss_dict["loss_loglikelihood"].item())
                # sums["loss_label_correlation"] += float(loss_dict["loss_label_correlation"].item())
                # sums["loss_graph"] += float(loss_dict["loss_graph"].item())

            denom = max(1, len(train_loader))
            train_avg = {k: v / denom for k, v in sums.items()}

            # -------- val --------
            val_avg = {
                "loss": None,
                "loss_kl": None,
                "loss_l1": None,
                "loss_impute": None,
                "loss_loglikelihood": None,
                # "loss_label_correlation": None,
                # "loss_graph": None
            }
            val_cos = None

            if val_loader is not None:
                self.model.eval()
                with torch.no_grad():
                    vsums = {
                        "loss": 0.0,
                        "loss_kl": 0.0,
                        "loss_l1": 0.0,
                        "loss_impute": 0.0,
                        "loss_loglikelihood": 0.0,
                        # "loss_label_correlation": 0.0,
                        # "loss_graph": 0.0
                    }
                    for xb, db in val_loader:
                        xb = xb.to(self.device)
                        db = db.to(self.device)
                        if self.model_type == 'linear':
                            pred_dist, R = self.model(xb)
                            loss_dict = self.criterion(pred_dist, db, R, modes="train_only", hidden=xb)
                        elif self.model_type == 'mlp':
                            pred_dist, R, hidden = self.model(xb)
                            loss_dict = self.criterion(pred_dist, db, R, modes="train_only", hidden=hidden)
                        vsums["loss"] += float(loss_dict["loss"].item())
                        vsums["loss_kl"] += float(loss_dict["loss_kl"].item())
                        vsums["loss_l1"] += float(loss_dict["loss_l1"].item())
                        vsums["loss_impute"] += float(loss_dict["loss_impute"].item())
                        vsums["loss_loglikelihood"] += float(loss_dict["loss_loglikelihood"].item())
                        # vsums["loss_label_correlation"] += float(loss_dict["loss_label_correlation"].item())
                        # vsums["loss_graph"] += float(loss_dict["loss_graph"].item())

                vdenom = max(1, len(val_loader))
                val_avg = {k: v / vdenom for k, v in vsums.items()}

                # distribution cosine on validation set
                if X_val_np is not None and D_val_np is not None:
                    D_val_pred = self.predict(X_val_np)
                    val_cos = self._cosine_np(D_val_np, D_val_pred)

                # best model / early-stopping update
                delta = float(early_stopping_min_delta)
                if best_metric == "loss":
                    current_score = float(val_avg["loss"])
                    improved = current_score < (best_score - delta)
                else:
                    current_score = float(val_cos if val_cos is not None else -float("inf"))
                    improved = current_score > (best_score + delta)

                if improved:
                    best_score = current_score
                    no_improve_epochs = 0
                    best_state = copy.deepcopy(
                        self.model.module.state_dict() if self.is_data_parallel else self.model.state_dict()
                    )
                    if best_model_path is not None:
                        best_model_path = str(best_model_path)
                        Path(best_model_path).parent.mkdir(parents=True, exist_ok=True)
                        torch.save(best_state, best_model_path)
                elif use_early_stopping:
                    no_improve_epochs += 1
                    if no_improve_epochs >= int(early_stopping_patience):
                        stop_training = True

                self.model.train()

            # Broadcast stop_training so all ranks exit the loop together
            if self.world_size > 1:
                stop_flag = torch.tensor([int(stop_training)], device=self.device)
                dist.broadcast(stop_flag, src=0)
                stop_training = bool(stop_flag.item())

            # -------- scheduler (all ranks use train_loss for consistency) --------
            if self.scheduler is not None:
                self.scheduler.step(float(train_avg["loss"]))
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            # -------- log / print (rank 0 only) --------
            if self.local_rank == 0:
                row = {
                    "epoch": epoch + 1,
                    "train_loss": train_avg["loss"],
                    "train_loss_kl": train_avg["loss_kl"],
                    "train_loss_l1": train_avg["loss_l1"],
                    "train_loss_impute": train_avg["loss_impute"],
                    "train_loss_loglikelihood": train_avg["loss_loglikelihood"],
                    "val_loss": val_avg["loss"],
                    "val_loss_kl": val_avg["loss_kl"],
                    "val_loss_l1": val_avg["loss_l1"],
                    "val_loss_impute": val_avg["loss_impute"],
                    "val_loss_loglikelihood": val_avg["loss_loglikelihood"],
                    "val_cosine": val_cos,
                    "best_metric": best_metric,
                    "best_score": best_score,
                    "lr": current_lr,
                }
                history.append(row)

                if csv_writer is not None:
                    csv_writer.writerow(row)
                    csv_file.flush()

                if verbose:
                    if val_loader is None:
                        print(
                            f"Epoch {epoch + 1}/{self.epochs} | "
                            f"loss={train_avg['loss']:.6f} "
                            f"(kl={train_avg['loss_kl']:.6f}, l1={train_avg['loss_l1']:.6f}, "
                            f"impute={train_avg['loss_impute']:.6f}, loglik={train_avg['loss_loglikelihood']:.6f}) "
                            f"| lr={current_lr:.2e}"
                        )
                    else:
                        val_cos_str = "None" if val_cos is None else f"{val_cos:.4f}"
                        print(
                            f"Epoch {epoch + 1}/{self.epochs} | "
                            f"train loss={train_avg['loss']:.6f} (kl={train_avg['loss_kl']:.6f}, l1={train_avg['loss_l1']:.6f}, "
                            f"impute={train_avg['loss_impute']:.6f}, loglik={train_avg['loss_loglikelihood']:.6f}) | "
                            f"val loss={val_avg['loss']:.6f} | val cosine={val_cos_str} | "
                            f"best({best_metric})={best_score:.6f} | lr={current_lr:.2e}"
                        )

            if stop_training:
                if verbose and self.local_rank == 0:
                    print(
                        f"Early stopping at epoch {epoch + 1}: "
                        f"{best_metric} has not improved for {int(early_stopping_patience)} epochs."
                    )
                break

        # restore best weights
        if restore_best and best_state is not None:
            if self.is_data_parallel:
                self.model.module.load_state_dict(best_state)
            else:
                self.model.load_state_dict(best_state)

        if csv_file is not None:
            csv_file.close()

        self.history_ = history
        return self

    def predict(self, X):
        """Return the simplex-normalized marginal mean (``eq:prediction``)."""
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        preds = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = X[i:i + self.batch_size]
                if self.model_type == 'linear':
                    pred_dist, R = self.model(xb)
                elif self.model_type == 'mlp':
                    pred_dist, R, hidden = self.model(xb)
                rank = torch.arange(
                    0, 1, 1.0 / self.num_bins,
                    device=xb.device,
                    dtype=pred_dist.dtype,
                ).view(1, -1)
                # Report the normalized marginal mean as the predicted label distribution.
                point = torch.matmul(rank, pred_dist).squeeze(1)
                point = torch.clamp(point, min=1e-8)
                point = point / point.sum(dim=1, keepdim=True)
                preds.append(point.cpu().numpy())

        return np.concatenate(preds, axis=0)

    def predict_with_uncertainty(self, X):
        """Return marginal means and dispersions for post-hoc association analysis.

        These dispersion outputs are not formal calibrated uncertainty estimates;
        ``train_uncertainty_eval.py`` evaluates only their association with error.
        """
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        self.model.eval()
        mu_raw_list = []
        mu_eval_list = []
        sigma_raw_list = []
        sigma_norm_list = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                xb = X[i:i + self.batch_size]
                if self.model_type == 'linear':
                    pred_dist, R = self.model(xb)
                elif self.model_type == 'mlp':
                    pred_dist, R, hidden = self.model(xb)

                rank = (
                    torch.arange(self.num_bins, device=xb.device, dtype=pred_dist.dtype)
                    / float(self.num_bins)
                ).view(1, -1)

                mu_raw = torch.matmul(rank, pred_dist).squeeze(1)
                mu_raw = torch.clamp(mu_raw, min=1e-8)
                denom = mu_raw.sum(dim=1, keepdim=True).clamp(min=1e-12)
                mu_eval = mu_raw / denom

                variance = torch.sum(
                    ((rank - mu_eval.reshape(mu_eval.shape[0], -1, 1)) ** 2)
                    * pred_dist.permute(0, 2, 1),
                    dim=2,
                )
                sigma_raw = torch.sqrt(torch.clamp(variance, min=1e-8))
                sigma_norm = sigma_raw / denom

                mu_raw_list.append(mu_raw.cpu().numpy())
                mu_eval_list.append(mu_eval.cpu().numpy())
                sigma_raw_list.append(sigma_raw.cpu().numpy())
                sigma_norm_list.append(sigma_norm.cpu().numpy())

        return {
            "mu_raw": np.concatenate(mu_raw_list, axis=0),
            "mu_eval": np.concatenate(mu_eval_list, axis=0),
            "sigma_raw": np.concatenate(sigma_raw_list, axis=0),
            "sigma_norm": np.concatenate(sigma_norm_list, axis=0),
        }
