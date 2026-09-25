"""
Multi-label validation script — matches training_tapfm_freezeN.py / training_tapfm.py.

Uses:
  - utils_tapfm.get_pfm()                for the tile encoder (same as training)
  - utils_tapfm.SlideSequentialDataset    for per-slide tile batching (same as training)
  - utils_tapfm.tapfm_aggregator          for the slide-level aggregator (same as training)
  - utils_tapfm.calculate_multilabel_metrics for macro AUC/FPR/FNR

Runs each tile checkpoint / slide checkpoint pair (paired by index) over the
validation split, computes per-slide multi-label predictions, and writes
macro AUC/FPR/FNR per checkpoint to a CSV.
"""

import os
import glob
import csv
import torch
import torch.nn.functional as F
import pandas as pd
import utils_tapfm


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_checkpoint(ckp_path, model, device, model_type='tile'):
    """Load checkpoint correctly based on model type"""
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(f"No checkpoint found at {ckp_path}")

    print(f"Loading checkpoint from {ckp_path}")
    checkpoint = torch.load(ckp_path, map_location=device)

    try:
        if model_type == 'tile':
            model.load_state_dict(checkpoint['tile_model'], strict=True)
        else:
            model.load_state_dict(checkpoint['slide_model'], strict=True)
        print(f"Successfully loaded {model_type} model from {ckp_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load {model_type} model: {str(e)}")

    return model


def ckpt_files(model_dir, filename_pattern):
    pattern = os.path.join(model_dir, str(filename_pattern + '_*.pth'))
    checkpoint_files = glob.glob(pattern)
    checkpoint_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    return checkpoint_files


# ---------------------------------------------------------------------------
# Forward pass helpers — identical logic to training_tapfm_freeze_lastN.py
# ---------------------------------------------------------------------------

def encoder_forward(tile_model, inputs):
    """Compute features using encoder (no autocast needed at eval; keep fp32)."""
    features = tile_model.forward(inputs)
    features = features.contiguous()
    features = features.float()
    return features


def get_cls_attention_weights_features(model, input_tensor, num_heads=24, head_dim=64):
    """Same hook-based cls-attention extraction used during training —
    kept identical so validation attention matches training attention."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    model_dir = ''
    pfm_name = 'gpath'
    tilesize = 224
    k_per_gpu = 1500  # tiles per slide at validation time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- target list must match training exactly (same columns, same order) ----
    df = pd.read_csv('')
    start_col = 23 # 6 for luad
    end_col = -3
    target_list = df.columns[start_col:end_col]
    print(df.columns[start_col:end_col].tolist())
    del df

    # ---- tile encoder (same construction path as training) ----
    tile_model = utils_tapfm.get_pfm(pfm_name)
    num_heads, head_dim, hidden_dim = utils_tapfm.get_head_info(tile_model)
    tile_model.ndim = hidden_dim

    # ---- dataset: multi-label, per-slide tile batching (same as training) ----


    dataset = utils_tapfm.SlideSequentialDataset(
        k=k_per_gpu,
        splits_csv_path='',
        tile_coords_csv_path='',
        slides_dir_path='',
        set_type = 'val',
        tilesize=tilesize,
        drop=0.0,
        target_list=target_list,
        rank=0) 

  
    dataset.makeData(1)
    loader = dataset.get_dataloader()

    print(f"Created validation dataset with {len(dataset)} slides")

    checkpoint_files_tile = ckpt_files(model_dir, 'checkpoint_tile')
    checkpoint_files_slide = ckpt_files(model_dir, 'checkpoint_slide')
    print(checkpoint_files_tile)
    print(checkpoint_files_slide)

    results_file = os.path.join(model_dir, 'validation_results_20x40x.csv')
    # with open(results_file, mode='w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(['epoch', 'macro_auc', 'macro_fpr', 'macro_fnr'])

    with open(results_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ['epoch', 'macro_auc', 'macro_fpr', 'macro_fnr']
        header += [f'{name}_auc' for name in target_list]
        writer.writerow(header)

    for idx in range(len(checkpoint_files_tile)):
        print(f"Loading Tile checkpoint: {checkpoint_files_tile[idx]}")
        tile_model = load_checkpoint(checkpoint_files_tile[idx], tile_model, device, model_type='tile')
        tile_model = tile_model.to(device)
        tile_model.eval()
        print("Tile Model Loaded.")

        slide_model = utils_tapfm.tapfm_aggregator(
            ndim=tile_model.ndim, n_classes=len(target_list), dropout_rate=0.50)
        print(f"Loading Slide checkpoint: {checkpoint_files_slide[idx]}")
        slide_model = load_checkpoint(checkpoint_files_slide[idx], slide_model, device, model_type='slide')
        slide_model = slide_model.to(device)
        slide_model.eval()
        print("Slide Model Loaded.")

        epoch_predictions = []
        epoch_labels = []

        with torch.no_grad():
            for slide_idx, tile_batches in loader:
                torch.cuda.empty_cache()
                label = dataset.get_target(slide_idx).to(device)

                all_features = []
                all_attention_weights = []
                for tile_batch in tile_batches:
                    features, attention_weights = get_cls_attention_weights_features(
                        tile_model, tile_batch.to(device))
                    all_features.append(features)
                    all_attention_weights.append(attention_weights)

                features = torch.cat(all_features, dim=0)
                attention_weights = torch.cat(all_attention_weights, dim=0)
                del all_features, all_attention_weights
                torch.cuda.empty_cache()

                # same attention normalization as training
                min_attn = attention_weights.min()
                max_attn = attention_weights.max()
                normalized_attn = (attention_weights - min_attn) / (max_attn - min_attn + 1e-8)
                attention_weights = F.softmax(normalized_attn / 0.1, dim=0)

                _, output = slide_model(features, attention_weights)
                torch.cuda.synchronize()
                prediction = torch.sigmoid(output).squeeze(0).detach().cpu()

                epoch_predictions.append(prediction)
                epoch_labels.append(label.detach().cpu())

                print(f"Slide {slide_idx + 1}/{len(loader)} processed")

        epoch_predictions = torch.stack(epoch_predictions).numpy()
        epoch_labels = torch.stack(epoch_labels).numpy()
        metrics = utils_tapfm.calculate_multilabel_metrics(epoch_predictions, epoch_labels, threshold=None)


        # inside the loop after metrics calculation
        per_class_aucs = [
            metrics['per_class'][f'class_{i}']['auc']
            for i in range(len(target_list))
            ]

        with open(results_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([idx, metrics['macro_auc'], metrics['macro_fpr'], metrics['macro_fnr']] + per_class_aucs)
       

        # with open(results_file, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow([idx, metrics['macro_auc'], metrics['macro_fpr'], metrics['macro_fnr']])

        print(f"Checkpoint {idx} - AUC: {metrics['macro_auc']:.4f} "
              f"- FPR: {metrics['macro_fpr']:.4f} - FNR: {metrics['macro_fnr']:.4f}")


if __name__ == '__main__':
    main()