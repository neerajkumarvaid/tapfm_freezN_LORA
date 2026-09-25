"""
diagnostic_choose_N.py
=======================================================================================

USAGE
  python diagnostic_choose_N.py \
      --splits_csv_path ...  --tile_coords_csv_path ...  --slides_dir_path ... \
      --outdir ./diagnostic_run --pfm gpath --calib_batches 16 --k_per_gpu 100 \
      --run_probe_crosschecks 1 --probe_steps 50
=======================================================================================
"""
 
import torch
import torch.nn.functional as F
import argparse
import copy
import os
import numpy as np
import pandas as pd
import utils_tapfm
 
try:
    from scipy.stats import spearmanr as _scipy_spearmanr
except Exception:
    _scipy_spearmanr = None
 
 
# ---------------------------------------------------------------------------
# Reused verbatim from training_tapfm_freeze_lastN_patchreduced.py so that
# the calibration/probe forward pass is identical to what training does.
# ---------------------------------------------------------------------------
def get_cls_attention_weights_features(model, input_tensor, num_heads=24, head_dim=64):
    """Unchanged from training_tapfm.py -- hooks the last block."""
    def hook_fn(module, input, output):
        qkv = module.qkv(input[0])
        B, N, _ = qkv.shape
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        cls_attn = attn[:, :, 0, 1:].mean(dim=1)
        image_importance = cls_attn.mean(dim=1, keepdim=True)
        return image_importance
 
    attention_result = None
 
    def hook_wrapper(module, input, output):
        nonlocal attention_result
        attention_result = hook_fn(module, input, output)
 
    last_block = model.blocks[-1]
    hook = last_block.attn.register_forward_hook(hook_wrapper)
    features = model(input_tensor)
    hook.remove()
    return features, attention_result
 
 
# ---------------------------------------------------------------------------
# Model setup: everything unfrozen. No optimizer for Stage 1 (never .step());
# real AdamW optimizers are only constructed in Stage 3's probe run.
# ---------------------------------------------------------------------------
def build_fully_unfrozen_encoder(args):
    """Loads the PFM with ALL parameters trainable (no freezing at all)."""
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        tile_model = utils_tapfm.get_pfm(args.pfm)
        num_heads, head_dim, hidden_dim = utils_tapfm.get_head_info(tile_model)
        tile_model.ndim = hidden_dim
        args.ndim = hidden_dim
        tile_model = tile_model.to(args.gpu)
 
    for param in tile_model.parameters():
        param.requires_grad = True
 
    n_blocks = len(tile_model.blocks)
    print(f"[diagnostic] Loaded {args.pfm} with {n_blocks} transformer blocks, ALL trainable.")
    return tile_model
 
 
def build_fresh_aggregator(args):
    """Freshly-initialized MIL aggregator + loss, per Algorithm 2's Input."""
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        slide_model = utils_tapfm.tapfm_aggregator(
            ndim=args.ndim, n_classes=len(args.target_list), dropout_rate=0.50)
        slide_model = slide_model.to(args.gpu)
 
    alpha_values = torch.tensor(args.alpha, dtype=torch.float)
    cw = torch.tensor(args.cw, dtype=torch.float)
    cw = cw * (len(args.target_list) / cw.sum())
    criterion = utils_tapfm.WeightedCrossEntropyMILLoss(class_weights=cw, alpha=alpha_values)
    return slide_model, criterion
 
 
# ---------------------------------------------------------------------------
# Mirrors of aggregator_step / encoder_backward, unified to optionally step
# a real optimizer (Stage 3 probe) or not (Stage 1 diagnostic, no-step).
# ---------------------------------------------------------------------------
def aggregator_forward_backward(slide_model, criterion, storage, dataset, batch_idx, args, optimizer=None):
    """optimizer=None -> pure measurement pass (Stage 1/2), never updates slide_model.
    optimizer=<AdamW> -> real training step (Stage 3 probe)."""
    features = storage.get_features().clone().detach().requires_grad_(True)
    attention_weights = storage.get_attention().clone().detach().requires_grad_(True)
 
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        _, output = slide_model(features, attention_weights)
        label = dataset.get_target(batch_idx).to(args.gpu)
        loss, _ = criterion(output, label)
 
    slide_model.zero_grad(set_to_none=True)
    loss.backward()
 
    storage.store_gradients(features.grad.detach())
    storage.store_attention_grads(attention_weights.grad.detach())
 
    if optimizer is not None:
        optimizer.step()
    return loss.item()
 
 
def encoder_taloss_backward(tile_model, features, attention_weights, storage, args, optimizer=None):
    """optimizer=None -> pure measurement pass (Stage 1/2), never updates tile_model.
    optimizer=<AdamW> -> real training step (Stage 3 probe). Either way, tile_model.*.grad
    is populated immediately after this call, so per-block RGN can be read off before the
    caller zeros gradients for the next iteration."""
    gradients = storage.get_gradients()
    attention_grads = storage.get_attention_grads()
 
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        scale_factor = 10.0
        tile_loss = (features * gradients).sum() * scale_factor
        aloss_innerproduct = (attention_weights * attention_grads).sum() * scale_factor
        total_loss = tile_loss + aloss_innerproduct
 
    tile_model.zero_grad(set_to_none=True)
    total_loss.backward()
 
    if optimizer is not None:
        optimizer.step()
    return total_loss.item()
 
 
# ---------------------------------------------------------------------------
# Per-block norms: Equations 6/7/8's ||theta_ell|| and ||grad_theta_ell L_PFM||.
# ---------------------------------------------------------------------------
def compute_block_weight_norms(tile_model):
    """w_ell = ||theta_ell||_2."""
    norms = []
    for block in tile_model.blocks:
        sq = sum((p.detach() ** 2).sum() for p in block.parameters())
        norms.append(float(sq.sqrt().item()))
    return np.array(norms)
 
 
def compute_block_grad_norms(tile_model):
    """||grad_theta_ell L_PFM||_2 for the CURRENT backward pass only.
    Call immediately after total_loss.backward() and before zeroing."""
    norms = []
    for block in tile_model.blocks:
        sq = 0.0
        for p in block.parameters():
            if p.grad is not None:
                sq += (p.grad.detach() ** 2).sum().item()
        norms.append(float(np.sqrt(sq)))
    return np.array(norms)
 
 
# ---------------------------------------------------------------------------
# Spearman correlation with a scipy-free fallback.
# ---------------------------------------------------------------------------
def spearman_corr(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if _scipy_spearmanr is not None:
        rho, p = _scipy_spearmanr(x, y)
        return float(rho), float(p)
    # Fallback: Pearson correlation of ranks; no p-value without scipy.
    rx = pd.Series(x).rank().values
    ry = pd.Series(y).rank().values
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho, float('nan')
 
 
# ---------------------------------------------------------------------------
# Linear CKA (Kornblith et al., 2019): CKA(X,Y) = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
# ---------------------------------------------------------------------------
def linear_cka(X, Y):
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    xty_f = np.linalg.norm(Y.T @ X, ord='fro')
    xtx_f = np.linalg.norm(X.T @ X, ord='fro')
    yty_f = np.linalg.norm(Y.T @ Y, ord='fro')
    return float((xty_f ** 2) / (xtx_f * yty_f + 1e-12))
 
 
# ---------------------------------------------------------------------------
# STAGE 1: Algorithm 2 calibration pass. Collects BOTH aggregation orders
# (mean-of-norms and mean-of-squared-norms == Fisher trace) per block from
# the SAME calibration batches, at no extra cost.
# ---------------------------------------------------------------------------
def run_calibration_pass(tile_model, slide_model, criterion, dataset, loader, args):
    n_blocks = len(tile_model.blocks)
    per_batch_norms = []  # list of [n_blocks] arrays, one per calibration WSI
 
    n_used = 0
    for slide_idx, tile_batches in loader:
        if n_used >= args.calib_batches:
            break
        torch.cuda.empty_cache()
 
        storage = utils_tapfm.SharedStorage()
        all_features, all_attn = [], []
        for tile_batch in tile_batches:
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                features, attention_weights = get_cls_attention_weights_features(
                    tile_model, tile_batch.to(args.gpu))
            all_features.append(features)
            all_attn.append(attention_weights)
 
        features = torch.cat(all_features, dim=0)
        attention_weights = torch.cat(all_attn, dim=0)
        del all_features, all_attn
 
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            min_attn = attention_weights.min()
            max_attn = attention_weights.max()
            normalized_attn = (attention_weights - min_attn) / (max_attn - min_attn + 1e-8)
            attention_weights = F.softmax(normalized_attn / 0.1, dim=0)
 
        storage.store_features(features)
        storage.store_attention(attention_weights)
 
        _ = aggregator_forward_backward(slide_model, criterion, storage, dataset, slide_idx, args, optimizer=None)
        _ = encoder_taloss_backward(tile_model, features, attention_weights, storage, args, optimizer=None)
 
        per_batch_norms.append(compute_block_grad_norms(tile_model))
        tile_model.zero_grad(set_to_none=True)
 
        n_used += 1
        print(f"[stage 1] calibration batch {n_used}/{args.calib_batches} (slide {slide_idx})")
 
    if n_used == 0:
        raise RuntimeError("No calibration batches were processed -- check your data paths / loader.")
 
    per_batch_norms = np.stack(per_batch_norms, axis=0)  # [n_used, n_blocks]
    mean_grad_norm = per_batch_norms.mean(axis=0)              # "mean-of-norms", Eq. 7 literal
    mean_grad_sq_norm = (per_batch_norms ** 2).mean(axis=0)    # "mean-of-squares" = Fisher trace, Eq. 6
    return mean_grad_norm, mean_grad_sq_norm, n_used
 
 
# ---------------------------------------------------------------------------
# STAGE 3: short real-AdamW probe run + representation hooks for CKA +
# weight delta-norm + AdamW v_t Fisher-trace readout.
# ---------------------------------------------------------------------------
def extract_block_representations(tile_model, tile_batches_list, args, max_tiles=None):
    """Registers a forward hook on every block, runs no-grad forward passes over
    the given list of tile batches, and returns {block_idx: [n_tiles, D] np.array}
    of mean-pooled (over tokens) per-block representations."""
    reps = {i: [] for i in range(len(tile_model.blocks))}
    handles = []
 
    def make_hook(idx):
        def hook(module, inp, out):
            pooled = out.detach().float().mean(dim=1)  # [B, D], pool over tokens
            reps[idx].append(pooled.cpu().numpy())
        return hook
 
    for idx, block in enumerate(tile_model.blocks):
        handles.append(block.register_forward_hook(make_hook(idx)))
 
    n_tiles_seen = 0
    with torch.no_grad():
        for tile_batch in tile_batches_list:
            if max_tiles is not None and n_tiles_seen >= max_tiles:
                break
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                tile_model(tile_batch.to(args.gpu))
            n_tiles_seen += tile_batch.shape[0]
 
    for h in handles:
        h.remove()
 
    return {i: np.concatenate(v, axis=0) for i, v in reps.items()}
 
 
def run_probe_and_crosschecks(tile_model, slide_model, criterion, dataset, loader, args):
    """Stage 3: (a) snapshots representations pre-probe, (b) runs a short REAL
    AdamW training probe, (c) snapshots representations post-probe, (d) computes
    per-block CKA drop, weight delta-norm, and an AdamW-v_t-derived Fisher-trace
    score. All quantities are for the RANK correlation with RGN (Section 4.4 /
    Appendix C.4), so no absolute-scale calibration of the AdamW score is needed."""
    n_blocks = len(tile_model.blocks)
 
    # ---- (a) collect a fixed held-out tile set for CKA, before touching weights ----
    print(f"[stage 3] collecting {args.cka_eval_wsis} held-out WSIs for CKA representation comparison")
    dataset.makeData(2)  # distinct sampling epoch from calibration (0) and probe (1)
    cka_loader = dataset.get_dataloader()
    cka_tile_batches = []
    n_wsi_collected = 0
    for slide_idx, tile_batches in cka_loader:
        if n_wsi_collected >= args.cka_eval_wsis:
            break
        cka_tile_batches.extend([tb.to(args.gpu) for tb in tile_batches])
        n_wsi_collected += 1
 
    reps_before = extract_block_representations(tile_model, cka_tile_batches, args, max_tiles=args.cka_max_tiles)
 
    # ---- weight snapshot before probe (CPU copy to save GPU memory) ----
    theta_before = {name: p.detach().cpu().clone() for name, p in tile_model.named_parameters()}
 
    # ---- (b) short real-AdamW probe run ----
    tile_optimizer = torch.optim.AdamW(tile_model.parameters(), lr=args.probe_lr, weight_decay=1e-4)
    slide_optimizer = torch.optim.AdamW(slide_model.parameters(), lr=args.probe_agg_lr, weight_decay=1e-4)
 
    dataset.makeData(1)  # distinct sampling epoch from calibration
    probe_loader = dataset.get_dataloader()
    n_used = 0
    for slide_idx, tile_batches in probe_loader:
        if n_used >= args.probe_steps:
            break
        torch.cuda.empty_cache()
 
        storage = utils_tapfm.SharedStorage()
        all_features, all_attn = [], []
        for tile_batch in tile_batches:
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                features, attention_weights = get_cls_attention_weights_features(
                    tile_model, tile_batch.to(args.gpu))
            all_features.append(features)
            all_attn.append(attention_weights)
 
        features = torch.cat(all_features, dim=0)
        attention_weights = torch.cat(all_attn, dim=0)
        del all_features, all_attn
 
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
            min_attn, max_attn = attention_weights.min(), attention_weights.max()
            normalized_attn = (attention_weights - min_attn) / (max_attn - min_attn + 1e-8)
            attention_weights = F.softmax(normalized_attn / 0.1, dim=0)
 
        storage.store_features(features)
        storage.store_attention(attention_weights)
 
        aggregator_forward_backward(slide_model, criterion, storage, dataset, slide_idx, args, optimizer=slide_optimizer)
        encoder_taloss_backward(tile_model, features, attention_weights, storage, args, optimizer=tile_optimizer)
        tile_model.zero_grad(set_to_none=True)
 
        n_used += 1
        print(f"[stage 3] probe step {n_used}/{args.probe_steps} (slide {slide_idx})")
 
    # ---- (c) post-probe representations on the SAME held-out tiles ----
    reps_after = extract_block_representations(tile_model, cka_tile_batches, args, max_tiles=args.cka_max_tiles)
 
    # ---- delta-norm per block ----
    block_param_names = [[] for _ in range(n_blocks)]
    for name in theta_before:
        for idx in range(n_blocks):
            if name.startswith(f"blocks.{idx}."):
                block_param_names[idx].append(name)
                break
 
    delta_norm = np.zeros(n_blocks)
    theta_after = dict(tile_model.named_parameters())
    for idx in range(n_blocks):
        num, den = 0.0, 0.0
        for name in block_param_names[idx]:
            before = theta_before[name]
            after = theta_after[name].detach().cpu()
            num += ((after - before) ** 2).sum().item()
            den += (before ** 2).sum().item()
        delta_norm[idx] = np.sqrt(num) / (np.sqrt(den) + 1e-12)
 
    # ---- CKA representation change per block ----
    cka_per_block = np.zeros(n_blocks)
    for idx in range(n_blocks):
        cka_per_block[idx] = linear_cka(reps_before[idx], reps_after[idx])
    one_minus_cka = 1.0 - cka_per_block
 
    # ---- AdamW v_t-derived Fisher trace per block (Appendix C.4) ----
    adamw_fisher = np.zeros(n_blocks)
    for idx, block in enumerate(tile_model.blocks):
        total = 0.0
        for p in block.parameters():
            state = tile_optimizer.state.get(p, None)
            if state is not None and 'exp_avg_sq' in state:
                total += state['exp_avg_sq'].detach().sum().item()
        adamw_fisher[idx] = total
 
    return delta_norm, one_minus_cka, adamw_fisher
 
 
# ---------------------------------------------------------------------------
# Elbow selection, shared by both score variants.
# ---------------------------------------------------------------------------
def make_elbow_fns(normalized_scores, n_blocks):
    def cumulative_score(N):
        if N <= 0:
            return 0.0
        return float(normalized_scores[n_blocks - N:].sum())
 
    def n_star(tau):
        for N in range(0, n_blocks + 1):
            if cumulative_score(N) >= tau:
                return N
        return n_blocks
 
    return cumulative_score, n_star
 
 
# ---------------------------------------------------------------------------
# Main orchestration. CKA (Stage 3) now runs BEFORE the elbow/N* computation
# and is used as the PRIMARY selection criterion; RGN/Fisher are kept as a
# free-to-compute reference/cross-check, not the primary recommendation.
# (Rationale: on real GigaPath calibration data, RGN's gradient-magnitude
# score can be dominated by high-variance outlier blocks -- e.g. block 0 --
# under class imbalance, whereas CKA's representation-change score does not
# show this failure mode and tracks brute-force-optimal N far more closely.)
# ---------------------------------------------------------------------------
def run_diagnostic(args):
    os.makedirs(args.outdir, exist_ok=True)
 
    tile_model = build_fully_unfrozen_encoder(args)
    slide_model, criterion = build_fresh_aggregator(args)
    n_blocks = len(tile_model.blocks)
 
    w = compute_block_weight_norms(tile_model)  # [L], fixed unless/until Stage 3 trains
 
    dataset = utils_tapfm.SlideSequentialDataset(
        k=args.k_per_gpu,
        splits_csv_path=args.splits_csv_path,
        tile_coords_csv_path=args.tile_coords_csv_path,
        slides_dir_path=args.slides_dir_path,
        tilesize=args.tilesize,
        drop=args.drop,
        target_list=args.target_list,
        rank=0)
    dataset.makeData(0)  # calibration uses a fixed sampling epoch, distinct from probe/CKA epochs
    loader = dataset.get_dataloader()
 
    # ================= STAGE 1: calibration pass, both aggregation orders =================
    mean_grad_norm, mean_grad_sq_norm, n_used = run_calibration_pass(
        tile_model, slide_model, criterion, dataset, loader, args)
 
    rgn_meanofnorms = mean_grad_norm / (w + 1e-12)              # Eq. 7, literal
    rgn_fisher = np.sqrt(mean_grad_sq_norm) / (w + 1e-12)       # sqrt(Tr F_ell)/||theta_ell||, Eq. 9 form
    fisher_trace = mean_grad_sq_norm                             # F_ell, Eq. 6
 
    s_rgn = rgn_meanofnorms / rgn_meanofnorms.sum()
    s_fisher = rgn_fisher / rgn_fisher.sum()
 
    df = pd.DataFrame({
        "block": np.arange(n_blocks),
        "weight_norm": w,
        "mean_grad_norm": mean_grad_norm,
        "mean_grad_sq_norm_fisher_trace": fisher_trace,
        "RGN_meanofnorms": rgn_meanofnorms,
        "RGN_fisher": rgn_fisher,
        "normalized_score_RGN": s_rgn,
        "normalized_score_Fisher": s_fisher,
    })
    df.to_csv(os.path.join(args.outdir, "rgn_per_block.csv"), index=False)
    print(f"[stage 1] wrote {os.path.join(args.outdir, 'rgn_per_block.csv')} "
          f"(averaged over {n_used} calibration WSIs)")
 
    cum_rgn, nstar_rgn = make_elbow_fns(s_rgn, n_blocks)
    cum_fisher, nstar_fisher = make_elbow_fns(s_fisher, n_blocks)
 
    # ================= STAGE 2: aggregation-order correlation (free) =================
    rho_agg_order, p_agg_order = spearman_corr(rgn_meanofnorms, rgn_fisher)
    crosscheck_rows = [{
        "comparison": "RGN_meanofnorms_vs_RGN_fisher (Appendix C.3, aggregation order)",
        "spearman_rho": rho_agg_order,
        "spearman_pvalue": p_agg_order,
    }]
    print(f"\n[stage 2] Spearman(RGN_meanofnorms, RGN_fisher) = {rho_agg_order:.4f} "
          f"(p={p_agg_order:.4g})  -- Appendix C.3")
 
    # ================= STAGE 3: probe-based cross-checks (CKA is now primary) =====
    s_cka, cum_cka, nstar_cka = None, None, None
    if args.run_probe_crosschecks:
        delta_norm, one_minus_cka, adamw_fisher = run_probe_and_crosschecks(
            tile_model, slide_model, criterion, dataset, loader, args)
 
        probe_df = pd.DataFrame({"block": np.arange(n_blocks), "delta_norm": delta_norm})
        probe_df.to_csv(os.path.join(args.outdir, "probe_delta_norm.csv"), index=False)
 
        cka_df = pd.DataFrame({
            "block": np.arange(n_blocks),
            "CKA_before_vs_after_probe": 1.0 - one_minus_cka,
            "one_minus_CKA": one_minus_cka,
        })
        cka_df.to_csv(os.path.join(args.outdir, "cka_representation_change.csv"), index=False)
 
        adamw_df = pd.DataFrame({"block": np.arange(n_blocks), "adamw_fisher_trace": adamw_fisher})
        adamw_df.to_csv(os.path.join(args.outdir, "adamw_fisher_trace.csv"), index=False)
 
        for name, arr in [("one_minus_CKA", one_minus_cka),
                           ("delta_norm", delta_norm),
                           ("adamw_fisher_trace", adamw_fisher)]:
            rho, p = spearman_corr(rgn_meanofnorms, arr)
            crosscheck_rows.append({
                "comparison": f"RGN_meanofnorms_vs_{name}",
                "spearman_rho": rho,
                "spearman_pvalue": p,
            })
            print(f"[stage 3] Spearman(RGN_meanofnorms, {name}) = {rho:.4f} (p={p:.4g})")
 
        print(f"\n[stage 3] wrote probe_delta_norm.csv, cka_representation_change.csv, "
              f"adamw_fisher_trace.csv to {args.outdir}")
 
        # ---- CKA-based elbow: the PRIMARY score for choosing N ----
        s_cka = one_minus_cka / (one_minus_cka.sum() + 1e-12)
        cum_cka, nstar_cka = make_elbow_fns(s_cka, n_blocks)
    else:
        print("\n[stage 3] skipped (--run_probe_crosschecks 0) -- "
              "no CKA score available, falling back to RGN for N* (not recommended, "
              "RGN can be unreliable under class imbalance; see script docstring).")
 
    # ================= N*(tau) table: CKA primary, RGN/Fisher kept for reference ====
    taus = [0.70, 0.80, 0.85, 0.90, 0.95]
    n_star_dict = {"tau": taus}
    if s_cka is not None:
        n_star_dict["N_star_CKA_PRIMARY"] = [nstar_cka(t) for t in taus]
        n_star_dict["cum_score_CKA_at_N_star"] = [cum_cka(nstar_cka(t)) for t in taus]
    n_star_dict["N_star_RGN_reference"] = [nstar_rgn(t) for t in taus]
    n_star_dict["cum_score_RGN_at_N_star"] = [cum_rgn(nstar_rgn(t)) for t in taus]
    n_star_dict["N_star_Fisher_reference"] = [nstar_fisher(t) for t in taus]
    n_star_dict["cum_score_Fisher_at_N_star"] = [cum_fisher(nstar_fisher(t)) for t in taus]
 
    n_star_df = pd.DataFrame(n_star_dict)
    n_star_df.to_csv(os.path.join(args.outdir, "N_star_by_tau.csv"), index=False)
    print("\n[N*] N*(tau); use N_star_CKA_PRIMARY for --unfreeze_last_n if Stage 3 ran:")
    print(n_star_df.to_string(index=False))
    if s_cka is not None:
        print(f"\n[N*] Recommended N* (CKA, primary, tau={args.tau}): {nstar_cka(args.tau)}")
    print(f"[N*] Reference N* (RGN, tau={args.tau}): {nstar_rgn(args.tau)}")
    print(f"[N*] Reference N* (Fisher, tau={args.tau}): {nstar_fisher(args.tau)}")
 
    # ---- real (measured) Figure 3: CKA primary curve, RGN/Fisher as reference ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
 
        cum_curve_rgn = [cum_rgn(N) for N in range(n_blocks + 1)]
        cum_curve_fisher = [cum_fisher(N) for N in range(n_blocks + 1)]
        primary_score = s_cka if s_cka is not None else s_rgn
        n_star_at_tau = nstar_cka(args.tau) if s_cka is not None else nstar_rgn(args.tau)
 
        fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.3))
 
        if s_cka is not None:
            axes[0].bar(np.arange(n_blocks), s_cka, width=0.7, color="#55a868",
                        label="1 - CKA (primary)")
        width = 0.35
        idx = np.arange(n_blocks)
        offset = width if s_cka is None else 0.0
        axes[0].bar(idx - offset, s_rgn, width=width, color="#4c72b0", alpha=0.6,
                    label="RGN (reference)")
        axes[0].set_xlabel("Block index (0 = input side, L-1 = output side)")
        axes[0].set_ylabel("Normalized score $\\tilde{s}_\\ell$")
        axes[0].set_title("(a) Per-block importance score (measured)")
        axes[0].legend(fontsize=7, frameon=False)
        axes[0].spines[["top", "right"]].set_visible(False)
 
        if s_cka is not None:
            cum_curve_cka = [cum_cka(N) for N in range(n_blocks + 1)]
            axes[1].plot(range(n_blocks + 1), cum_curve_cka, color="#55a868", linewidth=2.2,
                         label="1 - CKA (primary)")
        axes[1].plot(range(n_blocks + 1), cum_curve_rgn, color="#4c72b0", linewidth=1.2,
                     linestyle="--", alpha=0.7, label="RGN (reference)")
        axes[1].plot(range(n_blocks + 1), cum_curve_fisher, color="#c44e52", linewidth=1.0,
                     linestyle=":", alpha=0.7, label="Fisher-trace (reference)")
        axes[1].axhline(args.tau, color="gray", linestyle=":", linewidth=1)
        axes[1].axvline(n_star_at_tau, color="#dd8452", linestyle=":", linewidth=1.3)
        axes[1].text(n_star_at_tau + 0.5, 0.1, f"N*={n_star_at_tau}", fontsize=9, color="#dd8452")
        axes[1].set_xlabel("N (last-N blocks kept trainable)")
        axes[1].set_ylabel("Cumulative score $C(N)$")
        axes[1].set_title("(b) Cumulative-score elbow (measured)")
        axes[1].legend(fontsize=7, frameon=False)
        axes[1].spines[["top", "right"]].set_visible(False)
 
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "rgn_diagnostic.png"), dpi=200)
        plt.savefig(os.path.join(args.outdir, "rgn_diagnostic.pdf"))
        print(f"\n[fig] saved full debug comparison plot (CKA + RGN + Fisher overlay) to "
              f"{os.path.join(args.outdir, 'rgn_diagnostic.png')}")
    except Exception as e:
        print(f"[fig] debug plotting skipped ({e}); CSVs were still written.")
 
    # ---- PAPER FIGURE: clean CKA-only plot, no RGN/Fisher reference curves.        ----
    # ---- This is the figure that should go directly into the paper (Figure 3).    ----
    if s_cka is not None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
 
            n_star_at_tau = nstar_cka(args.tau)

            # n_star_at_tau = 10
            cum_curve_cka = [cum_cka(N) for N in range(n_blocks + 1)]
 
            fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))
 
            axes[0].bar(np.arange(n_blocks), s_cka, color="#55a868", width=0.8)
            axes[0].set_xlabel("Block index (0 = input side, L-1 = output side)")
            axes[0].set_ylabel("Normalized drift score $\\tilde{s}_\\ell$")
            axes[0].set_title("(a) Per-block representation drift (measured)")
            axes[0].spines[["top", "right"]].set_visible(False)
 
            axes[1].plot(range(n_blocks + 1), cum_curve_cka, color="#55a868", linewidth=2.0)
            axes[1].axhline(args.tau, color="gray", linestyle="--", linewidth=1)
            axes[1].axvline(n_star_at_tau, color="#dd8452", linestyle=":", linewidth=1.3)
            axes[1].text(n_star_at_tau + 0.5, 0.1, f"N*={n_star_at_tau}", fontsize=9, color="#dd8452")
            axes[1].set_xlabel("N (last-N blocks kept trainable)")
            axes[1].set_ylabel("Cumulative drift $C(N)$")
            axes[1].set_title("(b) Cumulative-drift elbow (measured)")
            axes[1].spines[["top", "right"]].set_visible(False)
 
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, "cka_diagnostic_paper.png"), dpi=200)
            plt.savefig(os.path.join(args.outdir, "cka_diagnostic_paper.pdf"))
            print(f"[fig] saved paper-ready Figure 3 (CKA only, no RGN/Fisher) to "
                  f"{os.path.join(args.outdir, 'cka_diagnostic_paper.png')}")
        except Exception as e:
            print(f"[fig] paper-figure plotting skipped ({e}); CSVs were still written.")
    else:
        print("[fig] paper-ready CKA figure skipped (--run_probe_crosschecks 0, no CKA data)")
 
    crosscheck_df = pd.DataFrame(crosscheck_rows)
    crosscheck_df.to_csv(os.path.join(args.outdir, "crosscheck_correlations.csv"), index=False)
    print(f"\n[done] wrote {os.path.join(args.outdir, 'crosscheck_correlations.csv')}")
    print(crosscheck_df.to_string(index=False))
 
    return df, n_star_df, crosscheck_df
 
 
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ---- same data/model args as training_tapfm_freeze_lastN_patchreduced.py ----
    parser.add_argument('--use_amp',    type=int,   default=1)
    parser.add_argument('--pfm',        type=str,   default='gpath')
    parser.add_argument('--target_list', nargs='+', default=['FGFR3_Binary'])
    parser.add_argument('--tilesize',   type=int,   default=224)
    parser.add_argument('--drop',       type=float, default=0.0)
    parser.add_argument('--splits_csv_path',      type=str, default=None)
    parser.add_argument('--tile_coords_csv_path', type=str, default=None)
    parser.add_argument('--slides_dir_path',      type=str, default=None)
    parser.add_argument('--gpu',        type=int,   default=0)
    parser.add_argument('--outdir',     type=str,   default=None)
    # ---- Stage 1 (calibration) args ----
    parser.add_argument('--calib_batches', type=int, default=100,
                         help='Number of calibration WSIs to average the per-block gradient '
                              'norm over (Algorithm 2, B).')
    parser.add_argument('--k_per_gpu', type=int, default=100,
                         help='Tiles per WSI. Must fit in memory with ALL blocks unfrozen -- '
                              'use the same budget as full TAPFM fine-tuning (e.g. 100 for '
                              'GigaPath), NOT your inflated post-freezing tile budget.')
    parser.add_argument('--tau', type=float, default=0.85,
                         help='Cumulative-score threshold for the single headline N* (Eq. 9).')
    # ---- Stage 3 (probe cross-checks) args ----
    parser.add_argument('--run_probe_crosschecks', type=int, default=1,
                         help='1 = also run the short real-AdamW probe for CKA / delta-norm / '
                              'AdamW-Fisher cross-checks (Section 4.4, Appendix C.4). '
                              '0 = only run Stage 1+2 (faster, no real weight updates at all).')
    parser.add_argument('--probe_steps', type=int, default=100,
                         help='Number of WSIs to train on during the short real-AdamW probe.')
    parser.add_argument('--probe_lr', type=float, default=1e-6,
                         help='PFM learning rate during the probe, matching TAPFM\'s recipe.')
    parser.add_argument('--probe_agg_lr', type=float, default=1e-5,
                         help='Aggregator learning rate during the probe, matching TAPFM\'s recipe.')
    parser.add_argument('--cka_eval_wsis', type=int, default=50,
                         help='Number of held-out WSIs used for the before/after CKA comparison.')
    parser.add_argument('--cka_max_tiles', type=int, default=2000,
                         help='Cap on total tiles pooled into the CKA representation matrices '
                              '(controls CPU memory / CKA compute cost).')
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.gpu = torch.device(f'cuda:{args.gpu}')
    else:
        args.gpu = torch.device('cpu')
    args.use_amp = bool(args.use_amp)
    args.run_probe_crosschecks = bool(args.run_probe_crosschecks)

     # blca
    args.outdir = ''
    args.splits_csv_path=''
    args.tile_coords_csv_path=''
    args.slides_dir_path=''
    
   


    # ---- class weights / alpha, computed the same way as the training script ----
    df_labels = pd.read_csv(args.splits_csv_path)
    start_col, end_col = 24, -3  # match training_tapfm_freeze_lastN_patchreduced.py; adjust if your CSV differs
    args.target_list = df_labels.columns[start_col:end_col]
    alpha, cw = [], []
    for col in df_labels.columns[start_col:end_col]:
        positives = df_labels[col].sum()
        total = len(df_labels)
        prop_pos = positives / total if total > 0 else 0
        alpha.append(1 - prop_pos)
        negatives = total - positives
        cw.append(negatives / positives if positives > 0 else 1)
    args.alpha = alpha
    args.cw = cw
    del df_labels

    args.tilesize = 224
    args.k_per_gpu = 100 
    args.optimizer = 'adamw'
    args.lr = 1e-06 
    args.warmup_epochs = 1 
    args.nepochs = 100 
    args.workers = 10 
    args.save_freq = 1 
    args.use_amp = bool(1)
    args.drop = 0.0 
    args.tau = 0.90

    
   
    run_diagnostic(args)
