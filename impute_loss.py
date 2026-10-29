"""Conditional-reconstruction and artificial-subset utilities for CORE-LDL.

Paper mapping: main labels ``eq:filled_target``,
``eq:conditional_reconstruction``, and ``eq:reconstruction_loss``; appendix
labels ``app:conditional_reconstruction`` and ``app:artificial_subsets``.

Mask convention:
    ``True`` in an artificial mask denotes a coordinate temporarily hidden and
    reconstructed. Its complement supplies conditioning coordinates. The active
    caller passes the detached filled target, so originally missing coordinates
    contain stop-gradient marginal means rather than ground truth.

Reported datasets use ``impute_loss_parallel`` with the fixed mask pool.
``impute_loss_dynamic_batch`` is the large-label fallback. Other reconstruction
implementations and optional regularizers are legacy utilities and are not part
of the reported four-term objective.
"""
import torch 
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
import csv
import copy
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset

def impute(target, R, mu):
    """Legacy Gaussian conditional-mean helper for one shared mask pattern."""
    batch_size = target.shape[0]
    mask = ~torch.isnan(target[0,:])
    o = torch.where(mask)[0]
    m = torch.where(~mask)[0]
    z = target - mu
    z_miss = z.clone()
    z_miss[:][:, m] = 0
    
    R_oo = R[:][:, o][..., o]  # [k, k]
    R_mo = R[:][:, m][..., o]  # [k, 6, k]
    
    # Stabilize the conditioning block before the legacy pseudoinverse.
    jitter = torch.eye(R_oo.size(-1), device=R_oo.device, dtype=R_oo.dtype) * 1e-4
    R_oo_stable = R_oo + jitter
    
    z_o = z[:][:, o].reshape(batch_size, o.shape[0], 1)
    R_oo_inv = torch.linalg.pinv(R_oo_stable)
    z_m = torch.matmul(torch.matmul(R_mo, R_oo_inv), z_o).reshape(batch_size, -1)
    
    z_miss[:][:, m] = z_m + mu[:, m]
    z_miss[:][:, o] = target[:][:, o]
    return z_miss

def impute_sigma(target, R, mu):
    """Legacy conditional-mean helper retained for the original loop path."""
    batch_size = target.shape[0]
    mask = ~torch.isnan(target[0,:])
    o = torch.where(mask)[0]
    m = torch.where(~mask)[0]
    z = target - mu
    z_miss = z.clone()
    z_miss[:][:, m] = 0
    
    R_oo = R[:][:, o][..., o]  
    R_mo = R[:][:, m][..., o]  
    
    # Stabilize the conditioning block before the legacy pseudoinverse.
    jitter = torch.eye(R_oo.size(-1), device=R_oo.device, dtype=R_oo.dtype) * 1e-4
    R_oo_stable = R_oo + jitter
    
    z_o = z[:][:, o].reshape(batch_size, o.shape[0], 1)
    R_oo_inv = torch.linalg.pinv(R_oo_stable)
    z_m = torch.matmul(torch.matmul(R_mo, R_oo_inv), z_o).reshape(batch_size, -1)
    
    z_miss[:][:, m] = z_m + mu[:, m]
    z_miss[:][:, o] = target[:][:, o]
    return z_miss

def generate_masks(num_labels=6, num_samples=63, *, use_full_mask_set=True, max_mask_set_size=None):
    """Build the fixed pool of nonzero artificial masks.

    Paper mapping: ``app:artificial_subsets``. Exhaustive mode creates all
    ``2**num_labels - 1`` nonzero masks. The all-label mask is generated here
    but filtered by reconstruction because its conditioning complement is empty.

    Args:
        num_labels: Number of label coordinates C.
        num_samples: Sample count when exhaustive enumeration is disabled.
        use_full_mask_set: Enumerate the complete nonzero mask pool.
        max_mask_set_size: Optional cap on the returned pool.

    Returns:
        Boolean tensor with shape ``(num_masks, num_labels)``.
    """
    # The all-zero mask creates no reconstruction target and is excluded.
    total_masks = 2 ** num_labels - 1

    nums = torch.arange(1, total_masks + 1, dtype=torch.int32)
    shifts = torch.arange(num_labels - 1, -1, -1)
    masks = ((nums.unsqueeze(1) >> shifts) & 1).to(torch.bool)

    if use_full_mask_set:
        if max_mask_set_size is not None and len(masks) > max_mask_set_size:
            idx = torch.randperm(len(masks))[:max_mask_set_size]
            masks = masks[idx]
    else:
        if num_samples is not None and len(masks) > num_samples:
            idx = torch.randperm(len(masks))[:num_samples]
            masks = masks[idx]

    return masks

def batch_convert_to_correlation_matrix(tensor, num_labels=14):
    """Expand strict-upper-triangle affinities as in ``eq:local_relation``."""
    n = tensor.shape[0]
    R = torch.zeros((n, num_labels, num_labels), dtype=tensor.dtype, device=tensor.device)
    rows, cols = torch.triu_indices(num_labels, num_labels, offset=1)
    R[:, rows, cols] = tensor
    R = R + R.transpose(1, 2)  
    R[:, torch.arange(num_labels), torch.arange(num_labels)] = 1.0
    return R

def find_duplicate_indices(tensor):
    """Legacy helper that groups identical masks; not used by reported runs."""
    tensor_int = tensor.to(dtype=torch.uint8)
    unique_rows, inverse_indices = torch.unique(tensor_int, return_inverse=True, dim=0)
    original_indices = torch.arange(tensor.size(0), device=tensor.device)
    groups = []
    for idx in range(unique_rows.size(0)):
        mask = (inverse_indices == idx)
        group = original_indices[mask]
        groups.append(group)
    return groups

def generate_random_masks_for_batch(batch_size, num_labels, masks_per_sample=4, missing_ratio_range=(0.3, 0.7)):
    """Sample artificial subsets for the large-label fallback.

    Each sample receives ``masks_per_sample`` nonempty masks while retaining at
    least one conditioning coordinate. Reported datasets have at most nine
    labels and therefore use the fixed exhaustive pool instead.

    Returns:
        Boolean tensor with shape ``(batch, masks_per_sample, labels)``.
    """
    masks = []
    
    for _ in range(batch_size):
        sample_masks = []
        for _ in range(masks_per_sample):
            min_ratio, max_ratio = missing_ratio_range
            n_missing = random.randint(
                max(1, int(num_labels * min_ratio)),
                max(1, int(num_labels * max_ratio))
            )
            n_missing = min(n_missing, num_labels - 1)
            
            if n_missing >= num_labels:
                n_missing = num_labels - 1
            
            missing_indices = random.sample(range(num_labels), n_missing)
            mask = torch.zeros(num_labels, dtype=torch.bool)
            mask[missing_indices] = True
            sample_masks.append(mask)
        masks.append(torch.stack(sample_masks))
    
    return torch.stack(masks)

def impute_loss_dynamic_batch(output, sigma, R, mu, target, masks_3d):
    """Vectorized conditional reconstruction for sample-specific masks.

    Paper mapping: ``eq:conditional_reconstruction`` and
    ``eq:reconstruction_loss``. The historical argument name ``R`` receives the
    full covariance Sigma from the caller. ``target`` is the detached filled
    target; any remaining NaNs indicate fill-disabled coordinates that cannot
    participate in the task.

    Returns:
        Mean absolute reconstruction error over valid artificial targets.
    """

    batch_size, masks_per_sample, num_labels = masks_3d.shape

    device = target.device



    # Flatten sample-mask pairs so every artificial task is processed in one batch.
    flat_masks = masks_3d.view(-1, num_labels) 

    flat_mu = mu.repeat_interleave(masks_per_sample, dim=0)

    flat_R = R.repeat_interleave(masks_per_sample, dim=0)

    flat_target = target.repeat_interleave(masks_per_sample, dim=0)

    

    # Separate conditioning coordinates from artificial reconstruction targets.
    actual_nan = torch.isnan(flat_target)

    obs_mask = (~actual_nan) & (~flat_masks)

    mis_mask = flat_masks & (~actual_nan)



    # Fixed-shape masking retains each conditioning covariance block and uses
    # identity padding elsewhere, avoiding variable-size submatrix extraction.
    jitter = torch.eye(num_labels, device=device).unsqueeze(0) * 1e-4

    

    eye_mask = torch.eye(num_labels, device=device).bool().unsqueeze(0).expand(batch_size*masks_per_sample, -1, -1)

    

    # Retain the conditioning block; padded coordinates receive identity entries.
    R_filtered = torch.where(obs_mask.unsqueeze(1) & obs_mask.unsqueeze(2), flat_R, torch.zeros_like(flat_R))

    R_filtered = torch.where(eye_mask, torch.ones_like(R_filtered), R_filtered) 

    

    # Solve Sigma_OO x = (target_O - mu_O) for the conditional correction.
    R_padded = R_filtered + jitter
    z_o = torch.where(obs_mask, flat_target-flat_mu, 0).unsqueeze(-1)  # (B*M, L, 1)
    R_inv_z_o = torch.linalg.solve(R_padded, z_o)                      # (B*M, L, 1)

    # Retain the cross-covariance Sigma_AO from conditioning to hidden coordinates.
    R_mo = torch.where(mis_mask.unsqueeze(2) & obs_mask.unsqueeze(1), flat_R, torch.zeros_like(flat_R))

    # Paper ``eq:conditional_reconstruction``: Sigma_AO Sigma_OO^{-1} residual_O.
    z_m = torch.bmm(R_mo, R_inv_z_o).squeeze(-1)

    

    # Add the marginal mean and average L1 error over artificial targets only.
    imputed_values = z_m + flat_mu

    

    diff = torch.abs(imputed_values - flat_target)

    loss_all = torch.where(mis_mask, diff, torch.zeros_like(diff))

    

    total_loss = loss_all.sum()

    total_valid = mis_mask.sum()



    return total_loss / (total_valid + 1e-8)

def impute_loss_dynamic(output, sigma, R, mu, loss_fn, target, masks_3d, mode):
    """Legacy loop-based dynamic reconstruction implementation.

    Reported runs use the vectorized ``impute_loss_dynamic_batch`` fallback
    instead. This function is retained only for implementation comparisons.
    """
    batch_size = target.shape[0]
    num_labels = target.shape[1]
    masks_per_sample = masks_3d.shape[1]
    device = target.device
    
    target_nan_mask = torch.isnan(target)
    
    total_loss = torch.tensor(0.0, device=device)
    total_valid = 0
    
    for b in range(batch_size):
        valid_obs = ~target_nan_mask[b]
        
        if valid_obs.sum() == 0:
            continue
        
        for i in range(masks_per_sample):
            mask_i = masks_3d[b, i]
            mis_idx = torch.where(mask_i)[0]
            
            if len(mis_idx) == 0:
                continue
            
            valid_for_impute = valid_obs & ~mask_i
            
            if valid_for_impute.sum() == 0:
                continue
            
            try:
                R_oo = R[b][valid_for_impute][:, valid_for_impute]
                R_mo = R[b][mis_idx][:, valid_for_impute]
                
                jitter = torch.eye(valid_for_impute.sum(), device=device) * 1e-4
                R_oo_stable = R_oo + jitter
                
                R_oo_inv = torch.linalg.pinv(R_oo_stable)
                
                z_o = (target[b, valid_for_impute] - mu[b, valid_for_impute]).reshape(-1, 1)
                z_m = torch.matmul(torch.matmul(R_mo, R_oo_inv), z_o).reshape(-1)
                
                imputed_values = z_m + mu[b, mis_idx]
                
                loss_val = torch.abs(imputed_values - target[b, mis_idx])
                
                valid_targets = ~torch.isnan(target[b, mis_idx])
                loss_val = loss_val[valid_targets]
                
                if len(loss_val) > 0:
                    total_loss += loss_val.sum()
                    total_valid += len(loss_val)
                    
            except:
                continue
    
    if total_valid == 0:
        return torch.tensor(0.0, device=device)
    
    return total_loss / total_valid

def impute_loss(output, sigma, R, mu, loss, tt, masks, mode):
    """Legacy grouped-mask reconstruction; not used by reported runs."""
    # Reference loop implementation retained for comparison with the parallel version.
    batch_size = output.shape[0]
    num_labels = tt.shape[1]
    groups = find_duplicate_indices(~torch.isnan(tt))
    imputeloss = torch.zeros((1,), device=output.device)  
    sum_val = 0  
    num_masks = masks.shape[0]
    
    for group in groups:
        mm = torch.isnan(tt[group, :])[0]
       
        if torch.any(mm):
            for i in range(num_masks):  
                mask = masks[i, :] 
                t = tt[group, :].clone()
                if torch.any(mm[mask]):
                    continue
                t[:, ~mask] = torch.nan
                t = impute_sigma(t, R[group, :, :], mu[group, :])
                
                valid_mask = ((~mask).int() * (~mm).int()).bool()
                imputeloss += loss(t[:, valid_mask], tt[group][:, valid_mask]).sum()
                sum_val += valid_mask.sum().item() * len(group)
           
        elif mode == 'train_only':
            for i in range(num_masks):
                mask = masks[i, :]
                t = tt[group, :].clone()
                t[:, ~mask] = torch.nan
                t = impute(t, R[group, :, :], mu[group, :])
                imputeloss += loss(t[:, ~mask], tt[group][:, ~mask]).sum()
                sum_val += (~mask).sum().item() * len(group)
                
    imputeloss /= (sum_val + 1e-8)
    return imputeloss

def impute_loss_parallel(output, sigma, R, mu, target, masks):
    """Evaluate fixed-pool conditional reconstruction in one vectorized batch.

    Paper mapping: ``eq:conditional_reconstruction``,
    ``eq:reconstruction_loss``, and ``app:conditional_reconstruction``. The
    historical argument name ``R`` receives the joint covariance Sigma. The
    target combines observed ground truth with detached marginal means at
    originally missing coordinates, so the latter provide structural targets
    without becoming direct supervision.
    """
    device = output.device
    batch_size = target.shape[0]
    num_masks, num_labels = masks.shape
    
    # Form every sample-mask pair so reconstruction can run in one batch.
    flat_masks = masks.repeat(batch_size, 1) 
    flat_target = target.repeat_interleave(num_masks, dim=0) 
    flat_mu = mu.repeat_interleave(num_masks, dim=0) 
    flat_R = R.repeat_interleave(num_masks, dim=0) 

    # NaNs remain only when detached filling is disabled for a coordinate.
    actual_nan = torch.isnan(flat_target)
    
    # A_i^c conditions the Gaussian operator; A_i supplies reconstruction targets.
    invalid_mask_comb = torch.any(actual_nan & flat_masks, dim=1) 
    obs_mask = (~actual_nan) & (~flat_masks)
    mis_mask = flat_masks & (~actual_nan)
    
    # Reject tasks that contain unavailable targets or no conditioning coordinate.
    is_invalid_sample = invalid_mask_comb | (obs_mask.sum(dim=1) < 1)
    
    # Reconstruction uses its own delta_rec=1e-2 stabilization, distinct from
    # the NLL diagonal shift tau.
    jitter_val = 1e-2 
    eye_batch = torch.eye(num_labels, device=device).unsqueeze(0).expand(batch_size * num_masks, -1, -1)
    
    # Retain each valid conditioning covariance block.
    safe_gate = (~is_invalid_sample).unsqueeze(1).unsqueeze(2) & obs_mask.unsqueeze(1) & obs_mask.unsqueeze(2)
    R_filtered = torch.where(safe_gate, flat_R, torch.zeros_like(flat_R))
    
    # Identity padding keeps all fixed-shape matrices numerically solvable.
    R_filtered = torch.where(eye_batch.bool(), torch.ones_like(R_filtered), R_filtered)
    
    # Apply the appendix safeguards before the Hermitian pseudoinverse.
    R_filtered = torch.clamp(R_filtered, -10, 10)
    
    # The cutoff discards numerically negligible eigen-directions.
    try:
        R_inv = torch.linalg.pinv(R_filtered + (eye_batch * jitter_val), rcond=1e-4, hermitian=True)
    except:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Paper ``eq:conditional_reconstruction``: covariance-conditioned correction.
    z_o = torch.where(obs_mask, flat_target-flat_mu, 0).unsqueeze(-1)
    R_mo_gate = (~is_invalid_sample).unsqueeze(1).unsqueeze(2) & mis_mask.unsqueeze(2) & obs_mask.unsqueeze(1)
    R_mo = torch.where(R_mo_gate, flat_R, torch.zeros_like(flat_R))
    
    # Clip only the reconstruction correction, as documented in the appendix.
    z_m = torch.bmm(torch.bmm(R_mo, R_inv), z_o).squeeze(-1)
    z_m = torch.clamp(z_m, -100, 100)
    
    # Paper ``eq:reconstruction_loss``: mean L1 over valid artificial targets.
    imputed_values = z_m + flat_mu
    diff = torch.abs(imputed_values - flat_target)
    
    # A target may be observed ground truth or a detached fill. Detached fills
    # can shape the reconstruction branch but cannot supervise their source.
    final_valid_mask = mis_mask & (~is_invalid_sample.unsqueeze(1))
    
    if not torch.any(final_valid_mask):
        return torch.tensor(0.0, device=device, requires_grad=True)

    loss_val = torch.where(final_valid_mask, diff, torch.zeros_like(diff)).sum()
    count = final_valid_mask.sum()

    return loss_val / (count + 1e-8)

def graph_loss(features, output, k=7, lambda2=0.01):
    """Legacy feature-graph regularizer, excluded from the reported objective.

    The helper constructs a Gaussian-weighted KNN graph from detached features
    and penalizes variation in flattened label-bin predictions over that graph.
    """
    m = features.shape[0]
    X = features.detach()

    # Construct a symmetric Gaussian-weighted KNN adjacency matrix.
    X_expand = X.unsqueeze(1)                           # (m,1,d)
    distance = torch.norm(X_expand - X, dim=2)          # (m,m)
    _, topk_idx = torch.topk(distance, k=k+1, dim=1, largest=False)
    topk_idx = topk_idx[:, 1:]                          # Exclude each sample itself.

    sigma = 1.0
    rows = torch.arange(m, device=features.device).unsqueeze(1).expand(m, k)  # (m,k)
    sim = torch.exp(-distance[rows, topk_idx] ** 2 / (2 * sigma ** 2))         # (m,k)

    A = torch.zeros((m, m), device=features.device)
    A[rows, topk_idx] = sim
    A = torch.max(A, A.T)

    # Form the normalized graph Laplacian.
    degree = torch.sum(A, dim=1)                        # (m,)
    D_inv_sqrt = torch.diag(1.0 / (degree.sqrt() + 1e-8))
    G = torch.eye(m, device=features.device) - D_inv_sqrt @ A @ D_inv_sqrt  # (m,m)

    # Flatten label-bin predictions before applying the graph trace penalty.
    output_flat = output.flatten(1)  # (B, 20 * num_labels)
    
    loss = lambda2 * torch.trace(output_flat.T @ G @ output_flat)

    return loss

def label_correlation_consistency_loss(R_raw, pre_prob, lambda3=0.1):
    """Legacy affinity-consistency penalty, disabled in reported CORE-LDL runs.

    The penalty weights squared pairwise differences between label-bin
    probabilities by the predicted affinity matrix.
    """
    batch_size = R_raw.shape[0]
    num_tokens = 20
    num_labels = pre_prob.shape[2]

    R_mat = batch_convert_to_correlation_matrix(R_raw, num_labels)

    # Form all pairwise label-probability differences.
    p_a = pre_prob.unsqueeze(3)  # (batch, 20, num_labels, 1)
    p_b = pre_prob.unsqueeze(2)  # (batch, 20, 1, num_labels)
    prob_diff_sq = (p_a - p_b) ** 2  # (batch, 20, num_labels, num_labels)

    # Broadcast each relation matrix over bins.
    R_mat = R_mat.unsqueeze(1)  # (batch, 1, num_labels, num_labels)

    # Exclude the diagonal and weight differences by predicted affinities.
    mask = 1 - torch.eye(num_labels, device=R_raw.device).unsqueeze(0).unsqueeze(0)  # (1,1,num_labels,num_labels)
    weighted = R_mat * prob_diff_sq * mask  # (batch,20,num_labels,num_labels)

    # Average over samples, bins, and ordered off-diagonal label pairs.
    loss = lambda3 * weighted.sum() / (batch_size * num_tokens * num_labels * (num_labels - 1))
    return loss
