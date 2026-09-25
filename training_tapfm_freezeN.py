"""
Layer-selection Strategy 1: Freeze all GigaPath blocks, train only the last N blocks.
=======================================================================================

USAGE:
  python training_tapfm_freeze_lastN.py --unfreeze_last_n 10 \
      --splits_csv_path ...  --tile_coords_csv_path ...  --slides_dir_path ... \
      --outdir ./run_lastN10  --pfm gpath
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
import utils_tapfm
import torch.nn.functional as F
import json
import numpy as np
import math
import pandas as pd


# ---------------------------------------------------------------------------
# The only function that differs from training_tapfm.py
# ---------------------------------------------------------------------------

def setup_encoder(args):
    """
    Setup encoder with selective freezing: only the last `args.unfreeze_last_n`
    transformer blocks (and the final LayerNorm) receive gradient updates.
    Everything else (patch_embed, cls_token, pos_embed, earlier blocks, head)
    is frozen before the optimizer is constructed, so those parameters never
    appear in any optimizer state and consume no extra memory.

    Freezing is done by setting requires_grad=False on ALL parameters first,
    then selectively re-enabling requires_grad=True on the last-N blocks +
    norm, BEFORE constructing the optimizer. This matters for two distinct
    reasons:
      1. (Optimizer/weight-update level) Frozen layers contribute zero
         gradient during encoder_backward's total_loss.backward(), so their
         weights never move, and the optimizer only tracks the un-frozen
         tensors.
      2. (Autograd-graph level, this is the memory win) Freezing the model
         top-to-bottom — not just tile_model.blocks — means requires_grad=
         False propagates cleanly from the very first op (patch_embed)
         through the frozen block prefix. Autograd then never builds/
         retains a backward graph for those frozen blocks at all, since an
         op's output only requires grad if at least one of its inputs or
         parameters does. That's what actually excludes the frozen 30
         blocks from the computational graph and frees up the activation
         memory needed to increase tiles per batch. (Freezing only
         tile_model.blocks while leaving patch_embed/cls_token/pos_embed
         trainable would NOT achieve this — the frozen blocks would still
         be pulled into the graph to backprop into those upstream params.)

    The aggregator_step and encoder_backward functions are unchanged:
    encoder_backward still calls tile_optimizer.step(), which silently
    does nothing for frozen layers because their grad is None.
    """
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        tile_model = utils_tapfm.get_pfm(args.pfm)
        num_heads, head_dim, hidden_dim = utils_tapfm.get_head_info(tile_model)
        tile_model.ndim = hidden_dim
        args.ndim = hidden_dim
        tile_model = tile_model.to(args.gpu)

    n_blocks = len(tile_model.blocks)

    # ------------------------------------------------------------------
    # Step 1: freeze the ENTIRE model (patch_embed, cls_token, pos_embed,
    # all blocks, norm, head — everything).
    #
    # IMPORTANT: freezing only tile_model.blocks (as before) is not enough
    # to remove frozen blocks from the computational graph. patch_embed /
    # cls_token / pos_embed sit *upstream* of block 0 and default to
    # requires_grad=True. As long as ANY upstream param requires grad,
    # autograd must build (and retain in memory) the full backward graph
    # for every block downstream of it too — even ones whose own weights
    # are frozen — just to be able to propagate gradients back to that
    # upstream param. So the first 30 blocks were still fully materialized
    # in the graph despite being "frozen", which is why you weren't seeing
    # the memory headroom to increase tiles per batch.
    #
    # Freezing everything first, then selectively unfreezing only the
    # last N blocks + norm, makes requires_grad=False propagate cleanly
    # from the very first op, so the frozen prefix is genuinely excluded
    # from the graph (forward-only, no activations retained for backward).
    # ------------------------------------------------------------------
    for param in tile_model.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Step 2: un-freeze the last N blocks (and norm) so they get updates
    # ------------------------------------------------------------------
    n_unfreeze = min(args.unfreeze_last_n, n_blocks)
    first_trainable = n_blocks - n_unfreeze   # e.g. 40-10 = 30 → blocks 30..39

    for i in range(first_trainable, n_blocks):
        for param in tile_model.blocks[i].parameters():
            param.requires_grad = True

    # Always keep the final LayerNorm trainable (cheap, but important)
    for param in tile_model.norm.parameters():
        param.requires_grad = True

    # Log which blocks are trainable
    frozen_count    = sum(1 for b in tile_model.blocks[:first_trainable]
                          for _ in b.parameters())
    trainable_count = sum(p.numel() for p in tile_model.parameters()
                          if p.requires_grad)
    total_count     = sum(p.numel() for p in tile_model.parameters())
    print(f"[freeze_lastN] Frozen: patch_embed/cls_token/pos_embed + "
          f"blocks 0..{first_trainable-1}  "
          f"| Trainable blocks: {first_trainable}..{n_blocks-1} + norm")
    print(f"[freeze_lastN] Trainable params: {trainable_count:,} / "
          f"{total_count:,} ({100*trainable_count/total_count:.1f}%)")

    # ------------------------------------------------------------------
    # Step 3: build optimizer ONLY over trainable params
    # ------------------------------------------------------------------
    params_groups = utils_tapfm.get_params_groups(tile_model)
    if args.optimizer == 'sgd':
        tile_optimizer = optim.SGD(
            params_groups, lr=0., momentum=args.momentum,
            dampening=0, nesterov=True)
    elif args.optimizer == 'adam':
        tile_optimizer = optim.Adam(params_groups)
    elif args.optimizer == 'adamw':
        tile_optimizer = optim.AdamW(params_groups, lr=args.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    print(f"Initialized encoder model with {sum(p.numel() for p in tile_model.parameters())} parameters")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        tile_optimizer, T_0=args.nepochs, T_mult=1, eta_min=1e-8)

    start_epoch, _ = utils_tapfm.restart_from_checkpoint(
        os.path.join(args.outdir, 'checkpoint_tile.pth'),
        tile_model=tile_model,
        tile_optimizer=tile_optimizer,
        scaler=scaler)

    return tile_model, tile_optimizer, scaler, scheduler, start_epoch



def setup_aggregator(args):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        slide_model = utils_tapfm.tapfm_aggregator(
            ndim=args.ndim, n_classes=len(args.target_list), dropout_rate=0.50)
        slide_model = slide_model.to(args.gpu)

    alpha_values = args.alpha
    alpha_values = torch.tensor(alpha_values, dtype=torch.float)
    cw = args.cw
    cw = torch.tensor(cw, dtype=torch.float)
    cw = cw * (len(args.target_list) / cw.sum())
    criterion = utils_tapfm.WeightedCrossEntropyMILLoss(class_weights=cw, alpha=alpha_values)
    args.pos_proportion = [1 - alpha for alpha in alpha_values]

    params_groups = utils_tapfm.get_params_groups(slide_model)
    if args.optimizer == 'sgd':
        slide_optimizer = optim.SGD(
            params_groups, lr=0., momentum=args.momentum, dampening=0, nesterov=True)
    elif args.optimizer == 'adam':
        slide_optimizer = optim.Adam(params_groups)
    elif args.optimizer == 'adamw':
        slide_optimizer = optim.AdamW(params_groups, lr=args.lr * 10)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        slide_optimizer, T_0=args.nepochs, T_mult=2, eta_min=1e-7)

    start_epoch, _ = utils_tapfm.restart_from_checkpoint(
        os.path.join(args.outdir, 'checkpoint_slide.pth'),
        slide_model=slide_model,
        slide_optimizer=slide_optimizer)
    print(f"Initialized aggregator model with {sum(p.numel() for p in slide_model.parameters())} parameters")
    return slide_model, slide_optimizer, criterion, scheduler, start_epoch


def encoder_forward(tile_model, inputs, args):
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        inputs = inputs.to(args.gpu)
        features = tile_model.forward(inputs)
        features = features.contiguous()
    features = features.float()
    return features


def aggregator_step(slide_model, slide_optimizer, criterion, storage, dataset, batch_idx, size_dset, args):
    """Run aggregator forward and backward pass — unchanged from training_tapfm.py"""
    features = storage.get_features()
    attention_weights = storage.get_attention()
    features = features.clone().detach().requires_grad_(True)
    attention_weights = attention_weights.clone().detach().requires_grad_(True)

    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        _, output = slide_model(features, attention_weights)
        torch.cuda.synchronize()
        prediction = torch.sigmoid(output)
        label = dataset.get_target(batch_idx).to(args.gpu)
        loss, class_loss = criterion(output, label)

    torch.cuda.synchronize()
    slide_optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()

    grad_norm = features.grad.norm().item()
    attention_grad_norm = attention_weights.grad.norm().item()

    storage.store_gradients(features.grad.detach())
    storage.store_attention_grads(attention_weights.grad.detach())
    torch.cuda.synchronize()
    slide_optimizer.step()
    torch.cuda.synchronize()
    return loss.item(), prediction, label


def encoder_backward(tile_model, tile_optimizer, scaler, features, attention_weights, storage, args):
    """Run encoder backward pass — unchanged from training_tapfm.py.
    Frozen layers get grad=None from autograd so tile_optimizer.step() skips them."""
    gradients = storage.get_gradients()
    attention_grads = storage.get_attention_grads()

    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
        scale_factor = 10.0
        tile_loss = (features * gradients).sum() * scale_factor
        aloss_innerproduct = (attention_weights * attention_grads).sum() * scale_factor
        total_loss = tile_loss + aloss_innerproduct

    tile_optimizer.zero_grad()
    total_loss.backward()
    tile_optimizer.step()

    return {'total_loss': total_loss.item(),
            'tile_loss': tile_loss.item(),
            'attention_loss': aloss_innerproduct.item()}


def get_cls_attention_weights_features(model, input_tensor, num_heads=24, head_dim=64):
    """Unchanged from training_tapfm.py — hooks the last block."""
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


def main(args):
    print(f"Training with args: {args}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tile_model, tile_optimizer, scaler, tile_scheduler, tsepoch = setup_encoder(args)
    slide_model, slide_optimizer, criterion, slide_scheduler, ssepoch = setup_aggregator(args)

    if tsepoch != ssepoch:
        print("Epoch mismatch between encoder and aggregator checkpoints")
        raise ValueError("Epoch mismatch between encoder and aggregator checkpoints")
    else:
        start_epoch = tsepoch
        print(f"Resuming training from epoch {start_epoch}")

    dataset = utils_tapfm.SlideSequentialDataset(
        k=args.k_per_gpu,
        splits_csv_path=args.splits_csv_path,
        tile_coords_csv_path=args.tile_coords_csv_path,
        slides_dir_path=args.slides_dir_path,
        tilesize=args.tilesize,
        drop=args.drop,
        target_list=args.target_list,
        rank=0)

    loader = dataset.get_dataloader()
    os.makedirs(args.outdir, exist_ok=True)

    for epoch in range(start_epoch, args.nepochs + 1):
        print(f"\nEpoch {epoch}/{args.nepochs}")
        dataset.makeData(epoch)

        running_loss = 0.0
        encoder_loss = 0.0
        epoch_predictions = []
        epoch_labels = []

        for slide_idx, tile_batches in loader:
            torch.cuda.empty_cache()

            label = dataset.get_target(slide_idx).to(device)
            storage = utils_tapfm.SharedStorage()

            all_features = []
            all_attention_weights = []

            for tile_batch in tile_batches:
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                    features, attention_weights = get_cls_attention_weights_features(
                        tile_model, tile_batch.to(args.gpu))
                all_features.append(features)
                all_attention_weights.append(attention_weights)

            features = torch.cat(all_features, dim=0)
            attention_weights = torch.cat(all_attention_weights, dim=0)
            del all_features, all_attention_weights
            torch.cuda.empty_cache()

            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=args.use_amp):
                min_attn = attention_weights.min()
                max_attn = attention_weights.max()
                normalized_attn = (attention_weights - min_attn) / (max_attn - min_attn + 1e-8)
                attention_weights = F.softmax(normalized_attn / 0.1, dim=0)

            storage.store_features(features)
            storage.store_attention(attention_weights)

            loss, prediction, _ = aggregator_step(
                slide_model, slide_optimizer,
                criterion, storage, dataset, slide_idx, len(loader), args)
            running_loss += loss

            epoch_predictions.append(prediction.squeeze(0).detach().cpu())
            epoch_labels.append(label.detach().cpu())

            eloss = encoder_backward(
                tile_model, tile_optimizer, scaler,
                features, attention_weights, storage, args)
            encoder_loss += eloss['total_loss']

            print(f"Slide {slide_idx+1}/{len(loader)} - "
                  f"Aggregator Loss: {loss:.4f} - Encoder Loss: {eloss['total_loss']:.7f}")

        avg_loss = running_loss / len(loader)
        avg_encoder_loss = encoder_loss / len(loader)
        epoch_predictions = torch.stack(epoch_predictions).numpy()
        epoch_labels = torch.stack(epoch_labels).numpy()
        metrics = utils_tapfm.calculate_multilabel_metrics(epoch_predictions, epoch_labels, threshold=None)

        metrics_path = os.path.join(args.outdir, f'metrics_epoch_{epoch}.json')
        with open(metrics_path, 'w') as f:
            json.dump(utils_tapfm.convert_to_serializable(metrics), f, indent=4)

        print(f"Epoch {epoch} - Aggregator Loss: {avg_loss:.4f} - Encoder Loss: {avg_encoder_loss:.4f}")
        print(f"Epoch {epoch} - AUC: {metrics['macro_auc']:.4f} - FPR: {metrics['macro_fpr']:.4f} - FNR: {metrics['macro_fnr']:.4f}")

        utils_tapfm.save_checkpoints(
            epoch, avg_loss, avg_encoder_loss, metrics,
            tile_model, slide_model, tile_optimizer, slide_optimizer, scaler, args)

        tile_scheduler.step()
        slide_scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_amp',    type=int,   default=1)
    parser.add_argument('--optimizer',  type=str,   default='adamw')
    parser.add_argument('--lr',         type=float, default=1e-6)
    parser.add_argument('--momentum',   type=float, default=0.9)
    parser.add_argument('--nepochs',    type=int,   default=20)
    parser.add_argument('--pfm',        type=str,   default='gpath')
    parser.add_argument('--target_list',nargs='+',  default=['FGFR3_Binary'])
    parser.add_argument('--k_per_gpu',  type=int,   default=100)
    parser.add_argument('--tilesize',   type=int,   default=224)
    parser.add_argument('--drop',       type=float, default=0.0)
    parser.add_argument('--splits_csv_path',      type=str, default=None)
    parser.add_argument('--tile_coords_csv_path', type=str, default=None)
    parser.add_argument('--slides_dir_path',      type=str, default='')
    parser.add_argument('--gpu',        type=int,   default=0)
    parser.add_argument('--outdir',     type=str,   default='./run_lastN')
    parser.add_argument('--save_freq',  type=int,   default=1)
    parser.add_argument('--outname',    type=str,   default='convergence.csv')
    # ---- Strategy-specific arg ----
    parser.add_argument('--unfreeze_last_n', type=int, default=10,
                        help='Number of GigaPath blocks (from the end) to keep trainable. '
                             '0 = freeze all PFM, 40 = train all (reproduces full TAPFM).')
    args = parser.parse_args()

    if torch.cuda.is_available():
        args.gpu = torch.device(f'cuda:{args.gpu}')
    else:
        args.gpu = torch.device('cpu')
    args.use_amp = bool(args.use_amp)

    df = pd.read_csv('')
    start_col = 23
    end_col = -3
    args.target_list = df.columns[start_col:end_col]
    print(df.columns[start_col:end_col].tolist())
    alpha = []
    cw = []
    for col in df.columns[start_col:end_col]:
        positives = df[col].sum()
        total = len(df)
        prop_pos = positives / total if total > 0 else 0
        alpha.append(1 - prop_pos)
        negatives = total - positives
        cw.append(negatives / positives if positives > 0 else 1)
   
    args.alpha = alpha
    args.cw = cw
    del df 

    args.tilesize = 224
    args.k_per_gpu = 250 
    args.optimizer = 'adamw'
    args.lr = 1e-06 
    args.warmup_epochs = 1 
    args.nepochs = 100 
    args.workers = 10 
    args.save_freq = 1 
    args.use_amp = bool(1)
    args.drop = 0.0 

    args.outdir = ''
    args.splits_csv_path=''
    args.tile_coords_csv_path=''
    args.slides_dir_path='A
    

    args.pfm = 'gpath'  # Use the GigaPath
    args.unfreeze_last_n = 10  # Freeze all but the last 20 blocks


    main(args)