import PIL.Image as Image
import openslide
from cucim import CuImage
import sys
import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import random
import torch
import torch.utils.data as data
from sklearn.metrics import roc_auc_score
import torchvision.transforms as transforms
import extra_transforms
import math
import pdb
import time
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import warnings
from torch import nn
import timm
from sklearn.metrics import roc_auc_score, roc_curve
import json
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import logging
import PIL.Image as Image
import openslide
import numpy as np
import pandas as pd
import random
import logging


def get_pfm(modelname):
    if modelname == 'gpath':
        return timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    elif modelname == 'hopt':
        return timm.create_model(
                "hf-hub:bioptimus/H-optimus-0", 
                pretrained=True, 
                init_values=1e-5, 
                dynamic_img_size=False
            )
    elif modelname == 'uni':
        return timm.create_model("hf-hub:MahmoodLab/uni", 
             pretrained=True, 
             init_values=1e-5, 
             dynamic_img_size=False)

def load_checkpoint(ckp_path, model, device, model_type='tile'):
    """Load checkpoint correctly based on model type"""
    if not os.path.isfile(ckp_path):
        raise FileNotFoundError(f"No checkpoint found at {ckp_path}")
        
    print(f"Loading checkpoint from {ckp_path}")
    checkpoint = torch.load(ckp_path, map_location=device, weights_only=False)
    
    try:
        if model_type == 'tile':
            # For tile model, load from 'tile_model' key
            model.load_state_dict(checkpoint['tile_model'], strict=True)
        else:
            # For slide model, load the state dict directly
            model.load_state_dict(checkpoint['slide_model'], strict=True)
        print(f"Successfully loaded {model_type} model from {ckp_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load {model_type} model: {str(e)}")
    
    return model

def ckpt_files(model_dir, filename_pattern):
      # Load the tile model
    # Pattern to match the desired checkpoint files
    pattern = os.path.join(model_dir, str(filename_pattern + '_*.pth'))

    # List all matching files
    checkpoint_files = glob.glob(pattern)

    # Sort the files by epoch number (assuming filenames include numeric epochs)
    checkpoint_files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1]))
    return checkpoint_files


# Add a custom JSON encoder to handle numpy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_timing_to_csv(epoch, timing_data, csv_path):
    """
    Save timing metrics to a CSV file.
    
    Args:
        epoch: Current epoch number
        timing_data: Dictionary containing timing metrics
        csv_path: Path to save the CSV file
    """
    # Check if file exists to determine if header is needed
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a') as f:
        # Create CSV writer
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            header = ['epoch', 'total_time', 'avg_time_per_wsi', 'std_time_per_wsi', 
                      'min_time', 'max_time', 'throughput']
            writer.writerow(header)
        
        # Write the data row
        row = [
            epoch,
            timing_data['total_time'],
            timing_data['avg_time_per_wsi'],
            timing_data['std_time_per_wsi'],
            timing_data['min_time'],
            timing_data['max_time'],
            timing_data['throughput']
        ]
        writer.writerow(row)
    
    print(f"Timing data for epoch {epoch} saved to {csv_path}")


def save_results(results_filename,epoch, target_list, metrics):
    
    # Ensure the file exists before writing
    if not os.path.exists(results_filename):
        # Create header with class-specific metrics
        header = 'epoch,macro_auc,macro_fpr,macro_fnr'

        # Add per-class metrics to header
        for class_name in target_list:
            # Remove '_Binary' suffix for cleaner headers
            clean_name = class_name.replace('_Binary', '')
            header += f',{clean_name}_auc,{clean_name}_fpr,{clean_name}_fnr'
        with open(results_filename, 'w') as fconv:
            fconv.write(header + '\n')  # Write header if file does not exist

    # Prepare data line
    data_line = f'{epoch},{metrics["macro_auc"]},{metrics["macro_fpr"]},{metrics["macro_fnr"]}'
    
    # Add per-class metrics to data line
    for i, class_name in enumerate(target_list):
        class_key = f'class_{i}'
        class_metrics = metrics['per_class'][class_key]
        data_line += f',{class_metrics["auc"]},{class_metrics["fpr"]},{class_metrics["fnr"]}'
    
    # Write data to file
    with open(results_filename, 'a') as fconv:
        fconv.write(data_line + '\n')


def load_thresholds_from_json(metric_file_path):
    """
    Load class-specific thresholds from a metric JSON file.
    
    Args:
        metric_file_path (str): Path to the metric JSON file
        
    Returns:
        list: List of threshold values for each class
    """
    with open(metric_file_path, 'r') as f:
        metric_data = json.load(f)
    
    # Extract class numbers and their corresponding thresholds
    thresholds = []
    class_indices = []
    
    for class_key, class_metrics in metric_data["per_class"].items():
        class_idx = int(class_key.split('_')[1])
        class_indices.append(class_idx)
        thresholds.append(class_metrics["threshold"])
    
    # Sort thresholds by class index
    sorted_thresholds = [t for _, t in sorted(zip(class_indices, thresholds))]
    
    return sorted_thresholds
def save_predictions_to_csv(epoch_predictions, epoch_labels, checkpoint_name, output_dir='results'):
    """
    Save predictions and labels to a CSV file.
    
    Parameters:
    - epoch_predictions: PyTorch tensor of model predictions
    - epoch_labels: PyTorch tensor of true labels
    - checkpoint_name: Name of the checkpoint (used in filename)
    - output_dir: Directory to save the CSV file (default: 'results')
    
    Returns:
    - Path to the saved CSV file
    """

    # Convert tensors to numpy if they're not already
    if isinstance(epoch_predictions, torch.Tensor):
        epoch_predictions = epoch_predictions.numpy()
    if isinstance(epoch_labels, torch.Tensor):
        epoch_labels = epoch_labels.numpy()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create column names based on the number of classes
    num_classes = epoch_predictions.shape[1]
    true_label_cols = [f'true_label_{i}' for i in range(num_classes)]
    pred_cols = [f'prediction_{i}' for i in range(num_classes)]
    binary_pred_cols = [f'binary_prediction_{i}' for i in range(num_classes)]
    
    # Create a DataFrame all at once
    results_dict = {
        **{true_label_cols[i]: epoch_labels[:, i] for i in range(num_classes)},
        **{pred_cols[i]: epoch_predictions[:, i] for i in range(num_classes)},
        **{binary_pred_cols[i]: (epoch_predictions[:, i] > 0.5).astype(int) for i in range(num_classes)}
    }
    results_df = pd.DataFrame(results_dict)
    
    # Save to CSV
    csv_filename = os.path.join(output_dir, f'predictions_checkpoint_{checkpoint_name}.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f"Saved results to {csv_filename}")

# Add this function to save timing results to CSV
def save_timing_to_csv(epoch, timing_data, csv_path):
    """
    Save timing metrics to a CSV file.
    
    Args:
        epoch: Current epoch number
        timing_data: Dictionary containing timing metrics
        csv_path: Path to save the CSV file
    """
    # Check if file exists to determine if header is needed
    file_exists = os.path.isfile(csv_path)
    
    with open(csv_path, 'a') as f:
        # Create CSV writer
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            header = ['epoch', 'total_time', 'avg_time_per_wsi', 'std_time_per_wsi', 
                      'min_time', 'max_time', 'throughput']
            writer.writerow(header)
        
        # Write the data row
        row = [
            epoch,
            timing_data['total_time'],
            timing_data['avg_time_per_wsi'],
            timing_data['std_time_per_wsi'],
            timing_data['min_time'],
            timing_data['max_time'],
            timing_data['throughput']
        ]
        writer.writerow(row)
    
    print(f"Timing data for epoch {epoch} saved to {csv_path}")

def get_head_info(model):
    # Get dimensions from the qkv projection in the last block
    last_block = model.blocks[-1]
    qkv_weight = last_block.attn.qkv.weight
    
    # Get hidden dimension (input features to qkv)
    hidden_dim = qkv_weight.shape[1]  # 1536 in your case
    
    # The output dimension is 3 times larger (for q, k, v)
    qkv_dim = qkv_weight.shape[0]     # 4608 in your case
    
    # Get number of heads from the model's config if available
    if hasattr(model, 'num_heads'):
        num_heads = model.num_heads
    else:
        # If not directly available, we can check module attributes
        attn_module = last_block.attn
        if hasattr(attn_module, 'num_heads'):
            num_heads = attn_module.num_heads
        else:
            # As a last resort, print dimensions for manual inspection
            print(f"Hidden dim: {hidden_dim}")
            print(f"QKV dim: {qkv_dim}")
            return None
    
    # Calculate head dimension
    head_dim = hidden_dim // num_heads
    
    return num_heads, head_dim, hidden_dim

def get_cls_attention_weights_ddp(model, input_tensor, num_heads=24, head_dim=64):
    """
    Optimized version of CLS token attention weight extraction with simplified structure.
    
    Args:
        model: Vision Transformer model
        input_tensor: Input tensor of shape [B, N, D]
        num_heads: Number of attention heads (default: 24)
        head_dim: Dimension of each attention head (default: 64)
    
    Returns:
        attention_weights: Tensor of shape [B, num_patches]
    """
    
    # Check if we're in distributed mode
    is_distributed = isinstance(model, DDP)
    if is_distributed:
        model = model.module
    
    # Tensors to store results
    attention_result = None
    attended_features_result = None
    
    def hook_fn(module, input, output):
        nonlocal attention_result, attended_features_result
        # QKV computation
        # input[0] shape: [B, N, D] where N is num_patches + 1 (CLS token)
        qkv = module.qkv(input[0])  # Shape: [B, N, 3*num_heads*head_dim]
        B, N, _ = qkv.shape
        
        # Reshape QKV
        qkv = qkv.reshape(B, N, 3, num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        
        # Unpack Q, K, V
        q, k, v = qkv.unbind(0)
        
        # Compute attention scores
        scale = head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        
        # Extract CLS token attention to patches (exclude CLS token attention to itself)
        cls_attention = attn[:, :, 0, 1:]  # [B, num_heads, num_patches]
        
        # Average across heads
        attention_result = cls_attention.mean(dim=1)  # [B, num_patches]

        # Compute attended features for CLS token
        # First get attended features for all tokens
        attended = attn @ v  # [B, num_heads, N, head_dim]
        # Extract CLS token's attended features
        cls_attended = attended[:, :, 0]  # [B, num_heads, head_dim]
        # Reshape to combine heads and head_dim
        attended_features_result = cls_attended.reshape(B, num_heads * head_dim)  # [B, num_heads * head_dim]   

    # Register hook on the last attention block
    last_block = model.blocks[-1]
    hook = last_block.attn.register_forward_hook(hook_fn)
    
    # Process input in chunks to manage memory
    chunk_size = 32
    with torch.no_grad():
        if input_tensor.size(0) > chunk_size:
            chunks = input_tensor.split(chunk_size)
            chunk_results = []
            features_chunks = []

            for chunk in chunks:
                # Ensure chunk is on the correct device
                if not chunk.is_cuda:
                    chunk = chunk.cuda()
                _ = model(chunk)
                chunk_results.append(attention_result)
                features_chunks.append(attended_features_result)

            # Concatenate all chunk results
            attention_result = torch.cat(chunk_results)
            attended_features_result = torch.cat(features_chunks)
        else:
            # Ensure input is on the correct device
            if not input_tensor.is_cuda:
                input_tensor = input_tensor.cuda()
            _ = model(input_tensor)
    
    hook.remove()
    
    # Handle distributed scenario
    if is_distributed:
        # Gather results from all GPUs
        gathered_weights = [torch.zeros_like(attention_result) 
                          for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_weights, attention_result)
        attention_result = torch.cat(gathered_weights)

        # Gather attended features
        gathered_features = [torch.zeros_like(attended_features_result) 
                           for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_features, attended_features_result)
        attended_features_result = torch.cat(gathered_features)

    # compute mean attention of raw attention weights
    mean_attention = torch.mean(attention_result, dim=1)
        
    # scale mean attention between 0 and 1
    min_attn = mean_attention.min()
    max_attn = mean_attention.max()
    mean_attention = (mean_attention - min_attn) / (max_attn - min_attn + 1e-8)
    
    attention_weights = F.softmax(mean_attention / 0.1, dim=0)
       
    attention_weights = attention_weights.view(-1, 1)    
    
    return attention_weights, attended_features_result

class tapfm_aggregator(nn.Module):
    """
    Simple attention-weighted aggregator for binary classification
    """
    def __init__(self, ndim, n_classes=2, dropout_rate=0.5):
        super().__init__()
        self.ndim = ndim
        
        # Classifier layers
    
        # Replace simple classifier with MLP head
        # self.classifier = nn.Sequential(
        #     nn.LayerNorm(ndim),
        #     nn.Linear(ndim, ndim // 2),
        #     nn.GELU(),
        #     nn.Dropout(dropout_rate),  # Direct instantiation,
        #     nn.Linear(ndim // 2, ndim // 4),
        #     nn.GELU(),
        #     nn.Dropout(dropout_rate),  # Direct instantiation
        #     nn.Linear(ndim // 4, n_classes)
        # )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(ndim),
            nn.Linear(ndim, n_classes)
        )
   

    def forward(self, features, attention_weights):
        """
        Forward pass computing bag features using attention weights
        
        Args:
            features: Feature vectors [N, D] where N is num_patches and D is feature dimension
            attention_weights: Attention weights [N] for each patch
        Returns:
            tuple: (weighted_features, logits)
                - weighted_features: Aggregated bag features [1, D]
                - logits: Classification logits [1, n_classes]
        """
    

        # Make sure attention_weights is the right shape before multiplying
        attention_weights = attention_weights.view(-1, 1)  # Reshape to [N, 1]
        # print(f"Attention weights shape: {attention_weights.shape}")

        # Verify shapes

        # Compute weighted features and sum
        # features: [N, 1536], attention_weights: [N, 1] -> broadcasts to [N, 1536]
        weighted_features = features * attention_weights  # [N, 1536]
    
        bag_features = weighted_features.sum(dim=0, keepdim=True)  # [1, 1536]
        
        # Get classification logits
        logits = self.classifier(bag_features)
        
        return bag_features, logits

# Setup logging for better error tracking
def setup_logging(log_file='slide_processing.log'):
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def validate_coordinates(slide_obj, x, y, size, level=0):
    """Validate if coordinates are within slide boundaries"""
    try:
        # Get slide dimensions at the specified level
        dimensions = slide_obj.level_dimensions[level]
        slide_width, slide_height = dimensions
        
        # Check if coordinates are within bounds
        if x < 0 or y < 0:
            return False, f"Negative coordinates: ({x}, {y})"
        
        if x + size > slide_width or y + size > slide_height:
            return False, f"Coordinates out of bounds: ({x}, {y}) + {size} exceeds slide dimensions {dimensions}"
        
        return True, "Valid coordinates"
        
    except Exception as e:
        return False, f"Error validating coordinates: {str(e)}"

def create_blank_image(size):
    """Create a blank white image as fallback"""
    blank_array = np.ones((size, size, 3), dtype=np.uint8) * 255  # White image
    return Image.fromarray(blank_array)

def safe_read_region(slide_obj, x, y, size, level=0, max_retries=3):
    """Safely read a region from slide with error handling and retries"""
    
    for attempt in range(max_retries):
        try:
            # Validate coordinates first
            is_valid, message = validate_coordinates(slide_obj, x, y, size, level)
            if not is_valid:
                raise ValueError(f"Invalid coordinates: {message}")
            
            # Try to read the region
            region = slide_obj.read_region(
                location=(int(x), int(y)), 
                size=(size, size), 
                level=level
            )
            
            # Convert to RGB array
            img_array = np.array(region)
            
            # Check if the image data is valid (not all zeros or corrupted)
            if img_array.size == 0:
                raise ValueError("Empty region returned")
            
            # Check for common corruption patterns
            if np.all(img_array == 0):
                raise ValueError("Region contains all zeros")
            
            # Convert to PIL Image and ensure RGB format
            img = Image.fromarray(img_array).convert('RGB')
            
            # Final validation - check if image can be processed
            img_test = np.array(img)
            if img_test.shape[0] == 0 or img_test.shape[1] == 0:
                raise ValueError("Invalid image dimensions after conversion")
            
            return img
            
        except openslide.lowlevel.OpenSlideError as e:
            if "Not a JPEG file" in str(e) or "starts with 0x00" in str(e):
                # This is likely a corrupted region, try alternative approach
                logging.warning(f"Attempt {attempt + 1}: Corrupted region at ({x}, {y}), size {size}: {str(e)}")
                
                # Try reading a smaller region or different coordinates
                if attempt < max_retries - 1:
                    # Slight offset to avoid corrupted area
                    x += 10
                    y += 10
                    size = max(size - 20, 50)  # Reduce size but keep minimum
                    continue
                else:
                    # Last attempt failed, return a blank image
                    logging.error(f"All attempts failed for region ({x}, {y}), returning blank image")
                    return create_blank_image(size)
            else:
                raise e
                
        except Exception as e:
            logging.warning(f"Attempt {attempt + 1}: Error reading region ({x}, {y}): {str(e)}")
            if attempt == max_retries - 1:
                # Last attempt, return blank image
                logging.error(f"All attempts failed for region ({x}, {y}), returning blank image")
                return create_blank_image(size)
            continue
    
    # Should not reach here, but just in case
    return create_blank_image(size)

class SlideCache:
    """LRU cache for slide objects with validation"""
    def __init__(self, slides_dir_path, max_slides=2):
        self.slides_dir_path = slides_dir_path
        self.max_slides = max_slides
        self.cache = {}  # Map slide_name to OpenSlide object
        self.lru_order = []  # Tracks least recently used slides
        self.invalid_slides = set()  # Track slides that failed to load
    
    def validate_slide(self, slide_path):
        """Validate slide file before loading"""
        try:
            # Check if file exists
            if not os.path.exists(slide_path):
                return False, f"Slide file not found: {slide_path}"
            
            # Try to open and get basic properties
            test_slide = openslide.OpenSlide(slide_path)
            dimensions = test_slide.dimensions
            level_count = test_slide.level_count
            test_slide.close()
            
            if dimensions[0] == 0 or dimensions[1] == 0:
                return False, f"Invalid slide dimensions: {dimensions}"
                
            return True, "Valid slide"
            
        except Exception as e:
            return False, f"Slide validation failed: {str(e)}"
    
    def get_slide(self, slide_name):
        """Get an OpenSlide object, loading if necessary"""
        if slide_name in self.invalid_slides:
            raise ValueError(f"Slide {slide_name} is marked as invalid")
            
        if slide_name in self.cache:
            # Update LRU order
            self.lru_order.remove(slide_name)
            self.lru_order.append(slide_name)
            return self.cache[slide_name]
        
        # Load the slide with validation
        slide_path = os.path.join(self.slides_dir_path, f'{slide_name}.svs')
        
        # Validate before loading
        is_valid, message = self.validate_slide(slide_path)
        if not is_valid:
            self.invalid_slides.add(slide_name)
            raise ValueError(f"Cannot load slide {slide_name}: {message}")
        
        try:
            slide_obj = openslide.OpenSlide(slide_path)
        except Exception as e:
            self.invalid_slides.add(slide_name)
            raise ValueError(f"Failed to load slide {slide_name}: {str(e)}")
        
        # Manage cache size
        if len(self.cache) >= self.max_slides:
            oldest = self.lru_order.pop(0)
            # Explicitly close the slide object to free resources
            try:
                self.cache[oldest].close()
            except:
                pass  # Ignore errors when closing
            del self.cache[oldest]
        
        # Add to cache
        self.cache[slide_name] = slide_obj
        self.lru_order.append(slide_name)
        
        return slide_obj
    
    def clear(self):
        """Clear all slides from cache"""
        for slide_name in list(self.cache.keys()):
            try:
                self.cache[slide_name].close()
            except:
                pass  # Ignore errors when closing
        self.cache = {}
        self.lru_order = []

class WSIDataloader:
    """Dataloader that processes one WSI at a time"""
    def __init__(self, 
                dataset,  # Reference to parent dataset
                batch_size=16,
                num_workers=4,
                pin_memory=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.current_slide_idx = -1
        self.current_loader = None
    
    def __iter__(self):
        self.current_slide_idx = -1
        return self
    
    def __next__(self):
        self.current_slide_idx += 1
        if self.current_slide_idx >= len(self.dataset.dfs):
            raise StopIteration
        
        # Create a single-slide dataloader
        self.dataset.prepare_slide(self.current_slide_idx)
        
        # Return slide index and tile batch
        tiles = self.dataset.get_slide_tiles()
        if self.batch_size < len(tiles):
            # Process in batches if there are too many tiles
            batches = []
            for i in range(0, len(tiles), self.batch_size):
                batch_tiles = tiles[i:i+self.batch_size]
                batches.append(default_collate(batch_tiles))
            return self.current_slide_idx, batches
        else:
            # Process all tiles at once if they fit in memory
            return self.current_slide_idx, [default_collate(tiles)]
    
    def __len__(self):
        return len(self.dataset.dfs)

class SlideSequentialDataset:
    """Memory-efficient dataset that processes one slide at a time with robust error handling"""
    def __init__(self, 
                 k,                          # Tiles per slide
                 splits_csv_path,            # Path to CSV with slide metadata
                 tile_coords_csv_path,       # Path to CSV with tile coordinates
                 slides_dir_path,            # Directory containing slide files
                 set_type='train',           # train/val/test
                 tilesize=224,               # Tile size in pixels
                 drop=0,                     # Fraction of slides to drop
                 seed=1634,                  # Random seed
                 target_list=None,           # List of target column names
                 rank=0,                     # Rank for distributed training
                 max_cached_slides=2):       # Maximum slides to keep in memory
        
        print("\nInitializing SlideSequentialDataset...")
        print(f"Reading splits from: {splits_csv_path}")
        
        # Read slide metadata
        self.dfs = pd.read_csv(splits_csv_path)
        nslides = int(len(self.dfs) * (1-drop))
        self.dfs = self.dfs.sample(n=nslides, random_state=seed).reset_index(drop=True)
        print(f"Total slides in manifest: {len(self.dfs)}")
        
        # Filter by set type
        print(f"\nFiltering for set_type: {set_type}")
        self.dfs = self.dfs[self.dfs.set==set_type].reset_index(drop=True)
        print(f"Slides after set filtering: {len(self.dfs)}")
        
        # Read tile coordinates
        print(f"\nReading tile coordinates from: {tile_coords_csv_path}")
        self.dft_master = pd.read_csv(tile_coords_csv_path)
        print(f"Total entries in tile coordinates: {len(self.dft_master)}")

        # Ensure slide_id columns are strings for matching
        self.dfs['slide_id'] = self.dfs['slide_id'].astype(str)
        self.dft_master['slide_id'] = self.dft_master['slide_id'].astype(str)

        # Filter slides with no tiles
        valid_slides = self.dfs[self.dfs.slide_id.isin(self.dft_master.slide_id)].slide_id.values
        if len(valid_slides) < len(self.dfs):
            print(f"\nWarning: Removing {len(self.dfs) - len(valid_slides)} slides with no tiles")
            self.dfs = self.dfs[self.dfs.slide_id.isin(valid_slides)].reset_index(drop=True)
        
        # Filter tile coordinates
        self.dft_master = self.dft_master[self.dft_master.slide_id.isin(self.dfs.slide_id)].reset_index(drop=True)
        
        print(f"\nRemaining slides after filtering: {len(self.dfs)}")
        print(f"Remaining tile entries: {len(self.dft_master)}")
        
        if len(self.dfs) == 0:
            raise ValueError("No valid slides remaining after filtering!")
        
        # Store parameters
        self.rank = rank
        self.k = k
        self.current_slide_idx = -1
        self.current_slide_tiles = None
        self.tilesize = tilesize
        self.seed = seed
        self.slides_dir_path = slides_dir_path
        self.target_list = target_list
        self.nslides = len(self.dfs)
        self.failed_slides = set()  # Track slides that consistently fail
        
        # Initialize slide cache with validation
        self.slide_cache = SlideCache(slides_dir_path, max_slides=max_cached_slides)
        
        # Initialize transforms
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            extra_transforms.RandomRectRotation(),
            extra_transforms.GaussianBlur(),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)) # hoptimus 
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) # Gpath and UNI
        ])
        
        # Create dataloader
        self.loader = WSIDataloader(self, batch_size=k//2)  # Process half k tiles at once
    
    def makeData(self, epoch):
        """Regenerate data for a new epoch"""
        np.random.seed(self.seed + epoch)
        random.seed(self.seed + epoch)
        
        # Reshuffle slides
        self.dfs = self.dfs.sample(frac=1, random_state=self.seed+epoch).reset_index(drop=True)
        
        # Clear slide cache
        self.slide_cache.clear()
        
        print(f"Prepared dataset for epoch {epoch} with {len(self.dfs)} slides")
    
    def prepare_slide(self, slide_idx):
        """Prepare tile data for a specific slide"""
        self.current_slide_idx = slide_idx
        row = self.dfs.iloc[slide_idx]
        
        # Get tile data for this slide
        slide_tiles = self.dft_master[self.dft_master.slide_id == row.slide_id]
        
        # Sample k tiles (or with replacement if fewer than k are available)
        if len(slide_tiles) >= self.k:
            slide_tiles = slide_tiles.sample(n=self.k, random_state=self.seed+slide_idx)
        else:
            slide_tiles = slide_tiles.sample(n=self.k, replace=True, random_state=self.seed+slide_idx)
        
        self.current_slide_tiles = slide_tiles
    
    def get_slide_tiles(self):
        """Get processed tiles for the current slide with robust error handling"""
        if self.current_slide_tiles is None:
            raise ValueError("No slide prepared. Call prepare_slide first.")
        
        row = self.dfs.iloc[self.current_slide_idx]
        slide_name = row.slide
        # slide_name = row.prefix + '/' + row.slide
        print(f"\nProcessing slide {self.current_slide_idx + 1}/{len(self.dfs)}: {slide_name}")
        
        # Skip slides that have consistently failed
        if slide_name in self.failed_slides:
            logging.warning(f"Skipping slide {slide_name} (marked as failed)")
            return [self._create_blank_tile() for _ in range(len(self.current_slide_tiles))]
        
        try:
            # Get slide object from cache
            slide_obj = self.slide_cache.get_slide(slide_name)
        except Exception as e:
            logging.error(f"Failed to load slide {slide_name}: {str(e)}")
            self.failed_slides.add(slide_name)
            return [self._create_blank_tile() for _ in range(len(self.current_slide_tiles))]
        
        # Process all tiles for this slide
        tiles = []
        failed_tiles = 0
        
        for i, tile_row in self.current_slide_tiles.iterrows():
            try:
                size = int(np.round(self.tilesize * tile_row.mult))
                
                # Use safe_read_region instead of direct read_region
                img = safe_read_region(
                    slide_obj, 
                    tile_row.x, 
                    tile_row.y, 
                    size, 
                    level=0
                )
                
                # Resize if necessary
                if tile_row.mult != 1:
                    img = img.resize((self.tilesize, self.tilesize), Image.LANCZOS)
                
                # Apply transformations
                img = self.transform(img)
                tiles.append(img)
                
            except Exception as e:
                logging.warning(f"Failed to process tile {i} from slide {slide_name}: {str(e)}")
                failed_tiles += 1
                
                # Create a blank tile as fallback
                blank_tile = self._create_blank_tile()
                tiles.append(blank_tile)
        
        if failed_tiles > 0:
            logging.warning(f"Slide {slide_name}: {failed_tiles}/{len(self.current_slide_tiles)} tiles failed")
            
            # If too many tiles failed, mark slide as problematic
            if failed_tiles / len(self.current_slide_tiles) > 0.5:
                logging.error(f"Slide {slide_name} has too many failed tiles, marking as failed")
                self.failed_slides.add(slide_name)
        
        return tiles
    
    def _create_blank_tile(self):
        """Create a blank tile with proper transformations applied"""
        blank_img = create_blank_image(self.tilesize)
        return self.transform(blank_img)
    
    def get_target(self, slide_idx):
        """Get target for a specific slide"""
        row = self.dfs.iloc[slide_idx]
        # Return a tensor with all binary target values
        target_vector = torch.tensor([int(row[col]) for col in self.target_list], dtype=torch.float)
        return target_vector
    
    def get_target_survival(self, slide_idx):
        """Get survival target for a specific slide"""
        row = self.dfs.iloc[slide_idx]
        # Return survival time and event status
        time = torch.tensor(float(row['Time']), dtype=torch.float)
        event = torch.tensor(int(row['Censor']), dtype=torch.long)
        return time, event
    
    def __len__(self):
        return len(self.dfs)
    
    def get_dataloader(self):
        """Get a dataloader that processes one slide at a time"""
        return self.loader
    
    def load_complete_survival_data(self):
        """Load all survival data for Cox loss computation"""
        survival_data = {
            'times': [],
            'events': [],
            'slide_ids': []
        }
    
        for idx in range(len(self.dfs)):
            row = self.dfs.iloc[idx]
            survival_data['times'].append(float(row['Time']))
            survival_data['events'].append(int(row['Censor']))
            survival_data['slide_ids'].append(row['slide_id'])
    
        # Convert to tensors
        survival_data['times'] = torch.tensor(survival_data['times'], dtype=torch.float)
        survival_data['events'] = torch.tensor(survival_data['events'], dtype=torch.long)
    
        return survival_data


def smooth_labels(targets, epsilon=0.1):
    """Apply label smoothing to targets"""
    return targets * (1 - epsilon) + epsilon / 2

class WeightedCrossEntropyMILLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for Multiple Instance Learning.
    Applies alpha weighting for positive/negative balance and class weights for multi-class combination.
    """
    def __init__(self, class_weights=None, alpha=None, eps=1e-6):
        """
        Args:
            class_weights: Fixed weights for each class based on dataset statistics
            alpha: Weighting factor for positive/negative balance (single value or per-class)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.class_weights = class_weights

        # Handle alpha - either single value or class-specific values
        if alpha is None:
            self.alpha = 0.25  # Default value if none provided
            self.class_specific_alpha = False
        elif isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
            self.class_specific_alpha = False
        else:
            # Convert to tensor if list/numpy array
            if isinstance(alpha, (list, tuple, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
            self.class_specific_alpha = True
        
    def forward(self, slide_prediction, slide_label):
        """
        Calculate weighted cross entropy loss on slide-level predictions.
        
        Args:
            slide_prediction: Model prediction for the whole slide, shape [1, num_classes] or [num_classes]
            slide_label: Ground truth label for the whole slide, shape [1, num_classes] or [num_classes]
            
        Returns:
            total_loss: Scalar loss value
            class_losses: Per-class loss values for monitoring
        """
        # Handle potential batch dimension (though in MIL this is typically 1)
        if len(slide_prediction.shape) > 1 and slide_prediction.shape[0] > 1:
            # If somehow we have a batch of slides, process each slide
            return self._compute_loss(slide_prediction, slide_label)
        
        # Ensure we have the right shapes
        if len(slide_prediction.shape) == 1:
            slide_prediction = slide_prediction.unsqueeze(0)
        if len(slide_label.shape) == 1:
            slide_label = slide_label.unsqueeze(0)
        
        return self._compute_loss(slide_prediction, slide_label)
    
    def _compute_loss(self, prediction, target):
        """Compute the weighted cross entropy loss"""
        # Apply label smoothing
        target = smooth_labels(target, epsilon=0.1) # use label smoothing if required otherwise comment this line

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(prediction)
        num_classes = prediction.shape[1]
        
        # Ensure alpha has correct format if class-specific
        if self.class_specific_alpha:
            if len(self.alpha) != num_classes:
                raise ValueError(f"Expected alpha to have {num_classes} elements, got {len(self.alpha)}")
            
            # Move alpha to the correct device
            if isinstance(self.alpha, torch.Tensor) and self.alpha.device != prediction.device:
                self.alpha = self.alpha.to(prediction.device)

        # Initialize tensor to hold per-class losses
        class_losses = torch.zeros(num_classes, device=prediction.device)
        
        # Process each class separately with its own alpha weighting
        for c in range(num_classes):
            # Get probabilities and targets for this class
            class_probs = probs[:, c]
            class_targets = target[:, c]

            # Get class-specific alpha if available, otherwise use global alpha
            if self.class_specific_alpha:
                if isinstance(self.alpha, torch.Tensor):
                    class_alpha = self.alpha[c].item()
                else:
                    class_alpha = self.alpha[c]
            else:
                class_alpha = self.alpha

            # Calculate binary cross entropy loss for this class
            # BCE = -[y*log(p) + (1-y)*log(1-p)]
            positive_loss = class_targets * torch.log(class_probs + self.eps)
            negative_loss = (1 - class_targets) * torch.log(1 - class_probs + self.eps)
            bce_loss = -(positive_loss + negative_loss)
            
            # Apply alpha weighting
            # For positive samples (y=1): weight = alpha
            # For negative samples (y=0): weight = (1-alpha)
            alpha_weight = class_targets * class_alpha + (1 - class_targets) * (1 - class_alpha)
            
            # Apply alpha weighting to BCE loss
            weighted_loss = alpha_weight * bce_loss
            
            # Take mean for this class
            class_losses[c] = weighted_loss.mean()
        
        # Apply class weights if provided
        if self.class_weights is not None:
            if isinstance(self.class_weights, (list, tuple, np.ndarray)):
                weights = torch.tensor(self.class_weights, device=class_losses.device, dtype=torch.float32)
            else:
                weights = self.class_weights.to(class_losses.device)
            
            class_losses = class_losses * weights
        
        # Return the summed loss and individual class losses
        return class_losses.sum(), class_losses

def get_model_size(model):
    """Calculate model size and gradient memory"""
    param_size = 0
    param_sum = 0
    grad_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()  # Memory used by parameters
        param_sum += param.nelement()
        
        # Memory used by gradients if they exist
        if param.grad is not None:
            grad_size += param.grad.nelement() * param.grad.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + grad_size + buffer_size) / 1024**2  # Convert to MB
    
    return {
        'total_size_mb': size_mb,
        'param_size_mb': param_size / 1024**2,
        'grad_size_mb': grad_size / 1024**2,
        'num_params': param_sum
    }

def get_params_groups(model1):
    regularized = []
    not_regularized = []
    for name, param in model1.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]


def restart_from_checkpoint(ckp_path, **kwargs):
    """
    Re-start from checkpoint
    """
    if not os.path.isfile(ckp_path):
        return 1, 0  # start from epoch 1, batch_file_idx 0
    print("Found checkpoint at {}".format(ckp_path))
    
    # open checkpoint file
    checkpoint = torch.load(ckp_path, map_location="cpu")
    
    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint '{}' with msg {}".format(key, ckp_path, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint: '{}'".format(key, ckp_path))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path))
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # batch_file_idx = checkpoint.get('batch_file_idx', 0)
    # Check for batch progress file
    
    outdir = osp.dirname(ckp_path)
    progress_path = osp.join(outdir, 'batch_progress.json')
    
    batch_file_idx = 0
    if os.path.isfile(progress_path):
        
        with open(progress_path, 'r') as f:
            progress = json.load(f)
            if progress['epoch'] == checkpoint['epoch']:
                batch_file_idx = progress['batch_file_idx']
                print(f"Found batch progress: will resume from batch {batch_file_idx}")

    # Return epoch start
    return checkpoint['epoch'], batch_file_idx


def save_checkpoints(epoch, loss, encoder_loss,metrics, tile_model, slide_model, 
                    tile_optimizer, slide_optimizer, scaler, args):
    """
    Save model checkpoints and training metrics.
    
    Args:
        epoch: Current epoch number
        loss: Average loss for the epoch
        encoder_loss: Average encoder loss for the epoch
        metrics: Dictionary containing metrics for each class
        tile_model, slide_model: Models to save
        tile_optimizer, slide_optimizer: Optimizers to save
        scaler: Gradient scaler to save
        args: Command line arguments containing save configuration
    
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    
    # Save training metrics
    # Ensure the file exists before writing
    filepath = os.path.join(args.outdir, args.outname)

    # Create header with class-specific metrics
    # header = 'epoch,aggregator_loss,encoder_loss,macro_auc,macro_fpr,macro_fnr'
    header = 'epoch,aggregator_loss,encoder_loss,macro_auc,micro_auc'

    # Add per-class metrics to header
    for class_name in args.target_list:
        # If any class name contains '_Binary', clean them; else use as is
        if any('_Binary' in name for name in args.target_list):
            clean_name = class_name.replace('_Binary', '')
        else:
            clean_name = class_name
        header += f',{clean_name}_auc'

    # Ensure the file exists before writing
    if not os.path.exists(filepath):
        with open(filepath, 'w') as fconv:
            fconv.write(header + '\n')  # Write header if file does not exist


    # Prepare data line
    # data_line = f'{epoch},{loss},{encoder_loss},{metrics["macro_auc"]},{metrics["macro_fpr"]},{metrics["macro_fnr"]}'
    data_line = f'{epoch},{loss},{encoder_loss},{metrics["macro_auc"]},{metrics["micro_auc"]}'

    # Add per-class metrics to data line
    for i, class_name in enumerate(args.target_list):
        class_key = f'class_{i}'
        class_metrics = metrics['per_class'][class_key]
        # data_line += f',{class_metrics["auc"]},{class_metrics["fpr"]},{class_metrics["fnr"]}'
        data_line += f',{class_metrics["auc"]}'
    # Write data to file
    with open(filepath, 'a') as fconv:
        fconv.write(data_line + '\n')

    # Save encoder checkpoint
    tile_obj = {
        'epoch': epoch + 1,
        'tile_model': tile_model.state_dict(),
        'tile_optimizer': tile_optimizer.state_dict(),
        'scaler': scaler.state_dict()
    }
    torch.save(tile_obj, os.path.join(args.outdir, 'checkpoint_tile.pth'))
    
    # Save periodic encoder checkpoint
    if epoch % args.save_freq == 0:
        torch.save(tile_obj, os.path.join(args.outdir, f'checkpoint_tile_{epoch:03}.pth'))
    
    # Save aggregator checkpoint
    slide_obj = {
        'epoch': epoch + 1,
        'slide_model': slide_model.state_dict(),
        'slide_optimizer': slide_optimizer.state_dict()
    }
    torch.save(slide_obj, os.path.join(args.outdir, 'checkpoint_slide.pth'))
    
    # Save periodic aggregator checkpoint
    if epoch % args.save_freq == 0:
        torch.save(slide_obj, os.path.join(args.outdir, f'checkpoint_slide_{epoch:03}.pth'))
    
    print(f"Saved checkpoints for epoch {epoch}")


def monitor_gradients(model, name):
    """Helper function to monitor gradient statistics"""
    total_norm = 0.0
    max_norm = 0.0
    min_norm = float('inf')
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            max_norm = max(max_norm, param_norm.item())
            min_norm = min(min_norm, param_norm.item())
    total_norm = total_norm ** 0.5
    print(f"{name} gradients - Total norm: {total_norm:.6f}, Max: {max_norm:.6f}, Min: {min_norm:.6f}")

def print_memory_stats():
    division_by = 1024**3 # change power to 2 to print in MBs, modify the print statements accordingly
    # Get memory values in GB (divide by 1024**3 to convert bytes to GB)
    total = torch.cuda.get_device_properties(0).total_memory / (division_by)
    allocated = torch.cuda.memory_allocated() / (division_by)
    cached = torch.cuda.memory_cached() / (division_by)
    available = total - cached
    
    print("\nGPU Memory Status:")
    print(f"Total GPU Memory: {total:.2f}GB")  # Total physical memory on your GPU
    print(f"Active Memory Use: {allocated:.2f}GB")  # Memory currently in use by tensors
    print(f"Reserved by PyTorch: {cached:.2f}GB")  # Total memory reserved by PyTorch
    print(f"Available Memory: {available:.2f}GB")  # Memory still free to use


class SharedStorage:
    """
    Storage class for sharing features and gradients between encoder and aggregator
    in single GPU training mode. This class provides a clean interface for data
    exchange without using distributed communication primitives.
    """
    def __init__(self, num_mutations=1):
        self.features = []  # Features flow from encoder to aggregator
        self.attention = []  # Attention weights flow from encoder to aggregator
        if num_mutations == 1:
            # Lists to store features and gradients for exchange
            self.gradients = []  # Gradients flow from aggregator to encoder
            self.attention_gradients = []  # Attention gradients flow from aggregator to encoder
        else:
            # Multiple mutation support
            self.feature_gradients = [[] for _ in range(num_mutations)]
            self.attention_gradients = [[] for _ in range(num_mutations)]
            self.loss_values = [0.0] * num_mutations
            self.num_mutations = num_mutations
        
    def store_features(self, features):
        """
        Store features computed by encoder for aggregator to use.
        
        Args:
            features (torch.Tensor): Computed features from encoder model
        """
        self.features.append(features.clone())
    
    def get_features(self):
        """
        Retrieve features for aggregator to process.
        
        Returns:
            torch.Tensor: Features computed by encoder
        """
        return self.features.pop(0)
    
    def store_gradients(self, gradients):
        """
        Store gradients computed by aggregator for encoder to use.
        
        Args:
            gradients (torch.Tensor): Computed gradients from aggregator's backward pass
        """
        self.gradients.append(gradients)
    
    def get_gradients(self):
        """
        Retrieve gradients for encoder to update its parameters.
        
        Returns:
            torch.Tensor: Gradients computed by aggregator
        """
        return self.gradients.pop(0)

    def store_attention(self, attention_weights):
        """
        Store features computed by encoder for aggregator to use.
        
        Args:
            features (torch.Tensor): Computed features from encoder model
        """
        self.attention.append(attention_weights.clone())
    
    def get_attention(self):
        """
        Retrieve features for aggregator to process.
        
        Returns:
            torch.Tensor: Features computed by encoder
        """
        return self.attention.pop(0)

    def store_attention_grads(self, attention_grads):
        """
        Store features computed by encoder for aggregator to use.
        
        Args:
            features (torch.Tensor): Computed features from encoder model
        """
        self.attention_gradients.append(attention_grads.clone())
    
    def get_attention_grads(self):
        """
        Retrieve features for aggregator to process.
        
        Returns:
            torch.Tensor: Features computed by encoder
        """
        return self.attention_gradients.pop(0)

     # New methods for multiple mutations
    def store_feature_gradients_list(self, gradients, mutation_idx):
        self.feature_gradients[mutation_idx].append(gradients.clone())
    
    def store_attention_gradients_list(self, gradients, mutation_idx):
        self.attention_gradients[mutation_idx].append(gradients.clone())
    
    def store_loss_value_list(self, loss_value, mutation_idx):
        self.loss_values[mutation_idx] = loss_value
    
    def get_feature_gradients_list(self):
        # Get the most recent gradient for each mutation
        gradients = [grads.pop(0) if grads else torch.zeros_like(self.features[0]) 
                    for grads in self.feature_gradients]
        return gradients
    
    def get_attention_gradients_list(self):
        # Get the most recent gradient for each mutation
        gradients = [grads.pop(0) if grads else torch.zeros_like(self.attention[0])
                    for grads in self.attention_gradients]
        return gradients
    
    def get_loss_values_list(self):
        return self.loss_values.copy()

    def clear(self):
        self.features.clear()
        self.gradients.clear()
        # Clear all lists
        if self.num_mutations == 1:
            self.attention.clear()
            slef.attention_gradients.clear()
        else:
             # Clear mutation-specific gradients
            for grads in self.feature_gradients:
                grads.clear()
            for grads in self.attention_gradients:
                grads.clear()
            # Reset loss values
            self.loss_values = [0.0] * self.num_mutations

        
        # Force CUDA cache clear if tensors were on GPU
        torch.cuda.empty_cache()

def calculate_multilabel_metrics(predictions, labels, threshold=None):
    """
    Calculate AUC, FPR, and FNR metrics for multi-label classification.
    
    Args:
        predictions (numpy.ndarray): Array of prediction probabilities with shape [n_samples, n_classes]
        labels (numpy.ndarray): Array of true binary labels with shape [n_samples, n_classes]
        threshold (float or list): Classification threshold(s) (default: None, will calculate optimal per class)
    
    Returns:
        dict: Dictionary containing metrics for each class and macro-averaged metrics
    """
    # Ensure inputs are numpy arrays
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Get number of classes
    n_classes = predictions.shape[1]
    
    # If threshold is a single value, use it for all classes
    if threshold is not None and not isinstance(threshold, list):
        thresholds = [threshold] * n_classes
    else:
        thresholds = [None] * n_classes
    # print(f"The thresholds are {thresholds}")
    # Initialize dictionaries to store per-class metrics
    # print(f"AUC-ROC of all classes: {roc_auc_score(labels, predictions, average=None)}")
    class_metrics = {}

    # Variables to store aggregated counts for micro-average calculation
    total_tp = 0
    total_fp = 0
    total_tn = 0
    total_fn = 0
    
    # Calculate metrics for each class
    for c in range(n_classes):
        class_preds = predictions[:, c]
        class_labels = labels[:, c]
        # print(f"Class {c} - Predictions size: {class_preds.size}, Labels: {class_labels.size}")
        # print(f"Class {c} - Predictions: {class_preds}, Labels: {class_labels}")

        # Skip classes with all zeros or all ones in labels (AUC is undefined)
        if np.all(class_labels == 0) or np.all(class_labels == 1):
            class_metrics[f'class_{c}'] = {
                'auc': float('nan'),
                'threshold': 0.5,  # default threshold
                'fpr': 0 if np.all(class_labels == 0) else 1.0,
                'fnr': 1.0 if np.all(class_labels == 0) else 0,
                'true_positives': 0,
                'false_positives': 0,
                'true_negatives': np.sum(class_labels == 0),
                'false_negatives': np.sum(class_labels == 1)
            }

            # Update totals for micro-average
            total_tn += np.sum(class_labels == 0)
            total_fn += np.sum(class_labels == 1)
            continue
        
        # Calculate AUC
        try:
            auc = roc_auc_score(class_labels, class_preds)
            # print(f"AUC-ROC of class {c}: {auc}")
        except ValueError:
            # Handle the case when there's only one class in the labels
            auc = float('nan')
        
        # Compute optimal threshold if not provided
        if thresholds[c] is None:
            fpr, tpr, th_values = roc_curve(class_labels, class_preds)
            # Compute the Youden's J statistic
            youden_j = tpr - fpr
            # Find the optimal threshold
            idx = np.argmax(youden_j)
            threshold_value = th_values[idx]
        else:
            threshold_value = thresholds[c]
        
        # Calculate binary predictions using threshold
        binary_preds = (class_preds >= threshold_value).astype(int)
        
        # Calculate confusion matrix elements
        false_positives = np.sum((binary_preds == 1) & (class_labels == 0))
        true_negatives = np.sum((binary_preds == 0) & (class_labels == 0))
        false_negatives = np.sum((binary_preds == 0) & (class_labels == 1))
        true_positives = np.sum((binary_preds == 1) & (class_labels == 1))

        # Update totals for micro-average
        total_tp += true_positives
        total_fp += false_positives
        total_tn += true_negatives
        total_fn += false_negatives
        
        # Calculate FPR and FNR
        fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0
        fnr = false_negatives / (false_negatives + true_positives) if (false_negatives + true_positives) > 0 else 0
        
        # Store metrics for this class
        class_metrics[f'class_{c}'] = {
            'auc': auc,
            'threshold': threshold_value,
            'fpr': fpr,
            'fnr': fnr,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives
        }
    
    # Calculate macro-average metrics
    valid_aucs = [metrics['auc'] for metrics in class_metrics.values() if not np.isnan(metrics['auc'])]
    macro_auc = np.mean(valid_aucs) if valid_aucs else float('nan')
    
    macro_fpr = np.mean([metrics['fpr'] for metrics in class_metrics.values()])
    macro_fnr = np.mean([metrics['fnr'] for metrics in class_metrics.values()])

    # Calculate micro-average metrics
    # Micro-averaged FPR and FNR
    micro_fpr = total_fp / (total_fp + total_tn) if (total_fp + total_tn) > 0 else 0
    micro_fnr = total_fn / (total_fn + total_tp) if (total_fn + total_tp) > 0 else 0
    
    # Micro-averaged AUC (using flattened predictions and labels)
    # Filter out classes that have all 0s or all 1s
    valid_indices = []
    for c in range(n_classes):
        class_labels = labels[:, c]
        if not (np.all(class_labels == 0) or np.all(class_labels == 1)):
            valid_indices.append(c)
    
    # Calculate micro-AUC if there are valid classes
    if valid_indices:
        try:
            # Flatten the valid predictions and labels for micro-AUC calculation
            flat_preds = predictions[:, valid_indices].flatten()
            flat_labels = labels[:, valid_indices].flatten()
            micro_auc = roc_auc_score(flat_labels, flat_preds)
        except ValueError:
            micro_auc = float('nan')
    else:
        micro_auc = float('nan')


    # Return combined metrics
    results = {
        'per_class': class_metrics,
        'macro_auc': macro_auc,
        'macro_fpr': macro_fpr,
        'macro_fnr': macro_fnr,
        'micro_auc': micro_auc,
        'micro_fpr': micro_fpr,
        'micro_fnr': micro_fnr,
        'confusion_matrix': {
            'total_true_positives': total_tp,
            'total_false_positives': total_fp,
            'total_true_negatives': total_tn,
            'total_false_negatives': total_fn
        }
    }
    
    return results

# Function to recursively convert NumPy types to native Python types
def convert_to_serializable(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def encoder_forward(tile_model, inputs):
    """Compute features using encoder"""
    features = tile_model.forward(inputs)
    features = features.contiguous()
    # Add debug print
        # print(f"Feature stats before float conversion - Min: {features.min():.6f}, Max: {features.max():.6f}")
    features = features.float()
    # Add debug print
    # print(f"Feature stats after float conversion - Min: {features.min():.6f}, Max: {features.max():.6f}")

    # Add normalization if needed
    # nfeatures = F.normalize(features, p=2, dim=1)  # L2 normalization
    # print(f"Feature stats after normalization - Min: {nfeatures.min():.6f}, Max: {nfeatures.max():.6f}")
    return features

def save_top_attention_tiles(
    slide_id,
    slide_path,
    all_features,
    all_attention_weights,
    tilecoords_to_load,
    true_labels,
    predictions,
    results_dir,
    class_idx=0, # position of the class label in mutation_list
    prob_high=0.90,
    prob_low=0.10
):
    """
    Save features and tile info for top 10% attention tiles if prediction meets criteria.
    """
    # Get class probability
    class_prob = predictions[0, class_idx]
    true_label = true_labels[class_idx]

    # if (class_prob > prob_high) or (class0_prob < prob_low):
    if (true_label == 1 and class_prob > 0.5) or (true_label == 0 and class_prob < 0.50):

        attn = all_attention_weights.cpu().numpy().flatten()
        num_top = max(1, int(0.1 * len(attn)))
        top_indices = np.argsort(attn)[-num_top:]

        selected_features = all_features[top_indices].cpu().numpy()
        selected_coords = tilecoords_to_load.iloc[top_indices].copy()
        selected_attn = attn[top_indices]

        # Save features
        feature_save_path = os.path.join(results_dir, f"{slide_id}_{class_prob}_top10pct_features.npy")
        np.save(feature_save_path, selected_features)

        # Save CSV with tile info
        selected_coords['slide_path'] = slide_path
        selected_coords['attention_value'] = selected_attn
        selected_coords['true_label'] = true_label
        selected_coords['pred_prob'] = class_prob

        selected_coords.to_csv(
            os.path.join(results_dir, f"{slide_id}_top10pct_tiles.csv"),
            index=False
        )
        print(f"Saved top 10% attention tiles for slide {slide_id} that has true_label {true_label} with class probability {class_prob}")
    else:
        print(f"Skipping slide {slide_id} - class probability {class_prob:.4f} does not meet criteria")


def save_top_attention_tiles_batch(
    slide_ids,
    slide_paths,
    all_features_list,
    all_attention_weights_list,
    tilecoords_list,
    true_labels_list,
    predictions_list,
    results_dir,
    class_idx=0,
    threshold=0.5
):
    """
    For a batch of slides:
    1. Find true positives and true negatives at a specific threshold.
    2. Among TPs, select those with highest 10% predicted probabilities.
    3. Among TNs, select those with lowest 10% predicted probabilities.
    4. For each selected slide, save tile features and attention scores for top 10% attention tiles.
    """
    # Convert lists to arrays for easier indexing
    slide_ids = np.array(slide_ids)
    slide_paths = np.array(slide_paths)
    true_labels = np.array([lbl[class_idx] for lbl in true_labels_list])
    pred_probs = np.array([pred[0, class_idx] for pred in predictions_list])
    pred_labels = (pred_probs > threshold).astype(int)

    # Step 1: Find TPs and TNs
    tp_mask = (true_labels == 1) & (pred_labels == 1)
    tn_mask = (true_labels == 0) & (pred_labels == 0)

    # Step 2: Among TPs, select top 10% by predicted probability
    tp_indices = np.where(tp_mask)[0]
    if len(tp_indices) > 0:
        num_tp_select = max(1, int(0.1 * len(tp_indices)))
        top_tp_indices = tp_indices[np.argsort(pred_probs[tp_indices])[-num_tp_select:]]
    else:
        top_tp_indices = []

    # Step 3: Among TNs, select bottom 10% by predicted probability
    tn_indices = np.where(tn_mask)[0]
    if len(tn_indices) > 0:
        num_tn_select = max(1, int(0.1 * len(tn_indices)))
        bottom_tn_indices = tn_indices[np.argsort(pred_probs[tn_indices])[:num_tn_select]]
    else:
        bottom_tn_indices = []

    selected_indices = np.concatenate([top_tp_indices, bottom_tn_indices])

    # Step 4: For each selected slide, save top 10% attention tiles
    for idx in selected_indices:
        slide_id = slide_ids[idx]
        slide_path = slide_paths[idx]
        all_features = all_features_list[idx]
        all_attention_weights = all_attention_weights_list[idx]
        tilecoords_to_load = tilecoords_list[idx]
        true_label = true_labels[idx]
        pred_label = pred_labels[idx]
        class_prob = pred_probs[idx]

        attn = all_attention_weights.cpu().numpy().flatten()
        num_top_tiles = max(1, int(0.1 * len(attn)))
        top_tile_indices = np.argsort(attn)[-num_top_tiles:]

        selected_features = all_features[top_tile_indices].cpu().numpy()
        selected_coords = tilecoords_to_load.iloc[top_tile_indices].copy()
        selected_attn = attn[top_tile_indices]

        # Save features
        selection_type = "TP" if true_label == 1 else "TN"
        feature_save_path = os.path.join(results_dir, f"{slide_id}_{selection_type}_top10pct_features.npy")
        np.save(feature_save_path, selected_features)

        # Save CSV with tile info
        selected_coords['slide_path'] = slide_path
        selected_coords['attention_value'] = selected_attn
        selected_coords['true_label'] = true_label
        selected_coords['pred_label'] = pred_label
        selected_coords['class_prob'] = class_prob

        selected_coords.to_csv(
            os.path.join(results_dir, f"{slide_id}_{selection_type}_top10pct_tiles.csv"),
            index=False
        )
        print(f"Saved {selection_type} top 10% attention tiles for slide {slide_id}")

class tapfm_cox_aggregator(nn.Module):
    """
    Cox proportional hazards aggregator for survival prediction
    """
    def __init__(self, ndim, dropout_rate=0.5):
        super().__init__()
        self.ndim = ndim
        
        # Cox regression typically outputs a single risk score (log hazard ratio)
        self.risk_predictor = nn.Sequential(
            nn.LayerNorm(ndim),
            nn.Linear(ndim, ndim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ndim // 2, ndim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ndim // 4, 1)  # Single risk score output
        )
   
    def forward(self, features, attention_weights):
        """
        Forward pass computing bag features using attention weights
        
        Args:
            features: Feature vectors [N, D] where N is num_patches and D is feature dimension
            attention_weights: Attention weights [N] for each patch
        Returns:
            tuple: (weighted_features, risk_score)
                - weighted_features: Aggregated bag features [1, D]
                - risk_score: Cox regression risk score [1, 1]
        """
        # Make sure attention_weights is the right shape
        attention_weights = attention_weights.view(-1, 1)  # Reshape to [N, 1]
        
        # Compute weighted features and sum
        weighted_features = features * attention_weights  # [N, D]
        bag_features = weighted_features.sum(dim=0, keepdim=True)  # [1, D]
        
        # Get risk score (log hazard ratio)
        risk_score = self.risk_predictor(bag_features)  # [1, 1]
        
        return bag_features, risk_score

class CoxPartialLikelihoodLoss1(nn.Module):
    """
    Cox Proportional Hazards partial likelihood loss for survival analysis
    """
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps
    
    def forward(self, risk_scores, times, events):
        """
        Calculate Cox partial likelihood loss
        
        Args:
            risk_scores: Model risk predictions [batch_size, 1]
            times: Survival times [batch_size]
            events: Event indicators (1=event, 0=censored) [batch_size]
            
        Returns:
            loss: Negative partial log-likelihood
        """
        # Flatten risk scores
        risk_scores = risk_scores.flatten()
        
        # Sort by survival time (descending order for Cox model)
        sorted_indices = torch.argsort(times, descending=True)
        sorted_times = times[sorted_indices]
        sorted_events = events[sorted_indices]
        sorted_risks = risk_scores[sorted_indices]
        
        # Calculate partial likelihood
        log_likelihood = 0.0
        
        for i in range(len(sorted_times)):
            if sorted_events[i] == 1:  # Only consider uncensored events
                # Risk set: all subjects still at risk at time t_i
                at_risk_mask = sorted_times >= sorted_times[i]
                risk_set_scores = sorted_risks[at_risk_mask]
                
                # Partial likelihood contribution
                numerator = sorted_risks[i]
                denominator = torch.logsumexp(risk_set_scores, dim=0)
                log_likelihood += numerator - denominator
        
        # Return negative log-likelihood as loss
        return -log_likelihood

def calculate_concordance_index(risk_scores, times, events):
    """
    Calculate Harrell's concordance index (C-index) for survival analysis
    
    Args:
        risk_scores: Array of risk predictions
        times: Array of survival times
        events: Array of event indicators (1=event, 0=censored)
        
    Returns:
        c_index: Concordance index value between 0 and 1
    """
    import numpy as np
    
    risk_scores = np.array(risk_scores).flatten()
    times = np.array(times).flatten()
    events = np.array(events).flatten()
    
    n = len(times)
    concordant = 0
    discordant = 0
    tied_risk = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            # Only consider pairs where we can make a meaningful comparison
            if times[i] != times[j]:
                # Determine which patient had the event first
                if times[i] < times[j]:
                    earlier_idx, later_idx = i, j
                else:
                    earlier_idx, later_idx = j, i
                
                # Only count if the earlier patient had an event
                if events[earlier_idx] == 1:
                    if risk_scores[earlier_idx] > risk_scores[later_idx]:
                        concordant += 1
                    elif risk_scores[earlier_idx] < risk_scores[later_idx]:
                        discordant += 1
                    else:
                        tied_risk += 1
    
    total_pairs = concordant + discordant + tied_risk
    if total_pairs == 0:
        return 0.5  # Random performance
    
    c_index = (concordant + 0.5 * tied_risk) / total_pairs
    return c_index



class CoxPartialLikelihoodLoss(nn.Module):
    """
    Cox Proportional Hazards partial likelihood loss using complete dataset context
    """
    def __init__(self, survival_data, eps=1e-7):
        super().__init__()
        self.eps = eps
        self.survival_times = survival_data['times']
        self.survival_events = survival_data['events']
        self.n_samples = len(self.survival_times)
    
    def forward(self, current_risk_score, current_time, current_event):
        """
        Calculate Cox partial likelihood using current sample and complete dataset context
        
        Args:
            current_risk_score: Risk score for current slide [1, 1]
            current_time: Survival time for current slide
            current_event: Event indicator for current slide
            
        Returns:
            loss: Negative partial log-likelihood
        """
        device = current_risk_score.device
        
        # Move survival data to appropriate device
        times = self.survival_times.to(device)
        events = self.survival_events.to(device)
        
        # Create risk scores: current sample gets predicted risk, others get zero baseline
        risk_scores = torch.zeros(self.n_samples + 1, device=device)
        risk_scores[0] = current_risk_score.flatten()
        
        # Create combined times and events
        combined_times = torch.cat([current_time.unsqueeze(0), times])
        combined_events = torch.cat([current_event.unsqueeze(0), events])
        
        # Sort by survival time (descending order)
        sorted_indices = torch.argsort(combined_times, descending=True)
        sorted_times = combined_times[sorted_indices]
        sorted_events = combined_events[sorted_indices]
        sorted_risks = risk_scores[sorted_indices]
        
        # Calculate partial likelihood
        log_likelihood = 0.0
        n_events = 0
        
        for i in range(len(sorted_times)):
            if sorted_events[i] == 1:  # Uncensored event
                # Risk set: all subjects still at risk at time t_i
                at_risk_mask = sorted_times >= sorted_times[i]
                risk_set_scores = sorted_risks[at_risk_mask]
                
                if len(risk_set_scores) > 1:
                    numerator = sorted_risks[i]
                    denominator = torch.logsumexp(risk_set_scores, dim=0)
                    log_likelihood += numerator - denominator
                    n_events += 1
        
        if n_events == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        return -log_likelihood / n_events

class SurvivalRegressionLoss(nn.Module):
    """
    Simplified survival loss that works effectively for individual samples
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, risk_score, time, event):
        """
        Calculate survival loss for individual sample
        
        Args:
            risk_score: Predicted risk score [1, 1]
            time: Survival time for current slide
            event: Event indicator (1=event, 0=censored)
        
        Returns:
            loss: Meaningful loss value for individual sample
        """
        risk_score = risk_score.flatten()
        
        if event == 1:  # Uncensored event
            # For events: encourage higher risk scores for shorter survival times
            # Loss increases as survival time decreases and risk score decreases
            normalized_time = torch.log(time + self.eps)
            loss = torch.exp(-risk_score) * normalized_time
        else:  # Censored
            # For censored: encourage lower risk scores for longer observed times
            # Loss increases as observed time increases and risk score increases
            normalized_time = torch.log(time + self.eps)
            loss = torch.exp(risk_score) / (normalized_time + self.eps)
        
        return loss.squeeze()

class IndividualCoxLoss(nn.Module):
    """
    Cox-inspired loss that provides meaningful gradients for individual samples
    """
    def __init__(self, baseline_risk=0.0, eps=1e-8):
        super().__init__()
        self.baseline_risk = baseline_risk
        self.eps = eps
    
    def forward(self, risk_score, time, event):
        """
        Calculate Cox-inspired loss using implicit baseline comparison
        """
        risk_score = risk_score.flatten()
        
        if event == 1:  # Uncensored event
            # Compare against baseline: log(exp(risk) / (exp(risk) + exp(baseline)))
            # Simplifies to: risk - log(exp(risk) + exp(baseline))
            baseline_exp = torch.exp(torch.tensor(self.baseline_risk, device=risk_score.device))
            risk_exp = torch.exp(risk_score)
            
            loss = -(risk_score - torch.log(risk_exp + baseline_exp + self.eps))
            
            # Weight by inverse survival time (shorter times should have higher risk)
            time_weight = 1.0 / (time + self.eps)
            loss = loss * time_weight
            
        else:  # Censored
            # Penalize high risk scores for censored cases
            # Encourages model to predict lower risk for longer observed times
            time_factor = torch.log(time + self.eps)
            loss = torch.abs(risk_score) / (time_factor + self.eps)
        
        return loss.squeeze()

def save_survival_checkpoints(epoch, aggregator_loss, encoder_loss, concordance_index, 
                              tile_model, slide_model, tile_optimizer, slide_optimizer, 
                              scaler, args):
    """
    Save model checkpoints and training metrics for survival analysis.
    
    Args:
        epoch: Current epoch number
        aggregator_loss: Average aggregator loss for the epoch
        encoder_loss: Average encoder loss for the epoch
        concordance_index: Calculated concordance index (C-index) for survival analysis
        tile_model, slide_model: Models to save
        tile_optimizer, slide_optimizer: Optimizers to save
        scaler: Gradient scaler to save
        args: Command line arguments containing save configuration
    """
    # Create output directory if it doesn't exist
    os.makedirs(args.outdir, exist_ok=True)
    
    # Save training metrics to CSV
    filepath = os.path.join(args.outdir, args.outname)
    
    # Create header for survival analysis metrics
    header = 'epoch,aggregator_loss,encoder_loss,concordance_index'
    
    # Create file with header if it doesn't exist
    if not os.path.exists(filepath):
        with open(filepath, 'w') as fconv:
            fconv.write(header + '\n')
    
    # Prepare and write data line
    data_line = f'{epoch},{aggregator_loss},{encoder_loss},{concordance_index}'
    with open(filepath, 'a') as fconv:
        fconv.write(data_line + '\n')
    
    # Save encoder checkpoint
    tile_checkpoint = {
        'epoch': epoch + 1,
        'tile_model': tile_model.state_dict(),
        'tile_optimizer': tile_optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'concordance_index': concordance_index,
        'aggregator_loss': aggregator_loss,
        'encoder_loss': encoder_loss
    }
    torch.save(tile_checkpoint, os.path.join(args.outdir, 'checkpoint_tile.pth'))
    
    # Save periodic encoder checkpoint
    if epoch % args.save_freq == 0:
        torch.save(tile_checkpoint, os.path.join(args.outdir, f'checkpoint_tile_{epoch:03}.pth'))
    
    # Save aggregator checkpoint
    slide_checkpoint = {
        'epoch': epoch + 1,
        'slide_model': slide_model.state_dict(),
        'slide_optimizer': slide_optimizer.state_dict(),
        'concordance_index': concordance_index,
        'aggregator_loss': aggregator_loss,
        'encoder_loss': encoder_loss
    }
    torch.save(slide_checkpoint, os.path.join(args.outdir, 'checkpoint_slide.pth'))
    
    # Save periodic aggregator checkpoint
    if epoch % args.save_freq == 0:
        torch.save(slide_checkpoint, os.path.join(args.outdir, f'checkpoint_slide_{epoch:03}.pth'))
    
    print(f"Saved survival model checkpoints for epoch {epoch}")
    # print(f"Concordance Index: {concordance_index:.4f}")


def save_survival_predictions_to_csv(epoch_risk_scores, epoch_times, epoch_events, 
                                   slide_ids, epoch, output_dir='results'):
    """
    Save survival predictions and labels to a CSV file.
    
    Parameters:
    - epoch_risk_scores: Array of predicted risk scores
    - epoch_times: Array of survival times
    - epoch_events: Array of event indicators (1=event, 0=censored)
    - slide_ids: List of slide identifiers
    - epoch: Current epoch number
    - output_dir: Directory to save the CSV file
    
    Returns:
    - Path to the saved CSV file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create DataFrame with survival data
    results_df = pd.DataFrame({
        'slide_id': slide_ids,
        'predicted_risk_score': epoch_risk_scores.flatten(),
        'survival_time': epoch_times.flatten(),
        'event_indicator': epoch_events.flatten(),
        'epoch': epoch
    })
    
    # Save to CSV
    csv_filename = os.path.join(output_dir, f'survival_predictions_epoch_{epoch}.csv')
    results_df.to_csv(csv_filename, index=False)
    print(f"Saved survival predictions to {csv_filename}")
    
    return csv_filename

class AFTLoss(nn.Module):
    """
    Accelerated Failure Time loss for individual slide computation
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, log_time_pred, time, event):
        """
        Calculate AFT loss for individual slide
        
        Args:
            log_time_pred: Predicted log survival time [1, 1]
            time: Actual survival time for current slide
            event: Event indicator (1=event, 0=censored)
        
        Returns:
            loss: AFT loss value for the slide
        """
        log_time_pred = log_time_pred.flatten()
        log_actual_time = torch.log(time + self.eps)
        
        if event == 1:  # Uncensored event
            # Standard regression loss for uncensored observations
            loss = F.mse_loss(log_time_pred, log_actual_time)
        else:  # Censored observation
            # Inequality constraint: predicted time should be >= observed time
            # Use a smooth approximation of max(0, log_actual_time - log_time_pred)
            diff = log_actual_time - log_time_pred
            loss = F.relu(diff) + 0.1 * torch.abs(diff)
        
        return loss

class WeibullAFTLoss(nn.Module):
    """
    Weibull-based AFT loss with parametric distribution modeling
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, distribution_params, time, event):
        """
        Calculate Weibull AFT loss
        
        Args:
            distribution_params: [log_scale, log_shape] parameters [1, 2]
            time: Actual survival time
            event: Event indicator
        """
        log_scale, log_shape = distribution_params.squeeze().unbind(0)
        scale = torch.exp(log_scale)
        shape = torch.exp(log_shape)
        
        if event == 1:  # Uncensored
            # Weibull log-likelihood
            log_pdf = (torch.log(shape + self.eps) - log_scale + 
                      (shape - 1) * torch.log(time + self.eps) - 
                      torch.pow(time / scale, shape))
            loss = -log_pdf
        else:  # Censored
            # Survival function log-likelihood
            log_survival = -torch.pow(time / scale, shape)
            loss = -log_survival
        
        return loss

class tapfm_aft_aggregator(nn.Module):
    """
    Accelerated Failure Time aggregator for direct survival time prediction
    """
    def __init__(self, ndim, dropout_rate=0.5):
        super().__init__()
        self.ndim = ndim
        
        # AFT regression predicts log survival time
        self.time_predictor = nn.Sequential(
            nn.LayerNorm(ndim),
            nn.Linear(ndim, ndim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ndim // 2, ndim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ndim // 4, 1)  # Single log time output
        )
   
    def forward(self, features, attention_weights):
        """
        Forward pass computing bag features and predicted log survival time
        
        Args:
            features: Feature vectors [N, D] where N is num_patches and D is feature dimension
            attention_weights: Attention weights [N] for each patch
        Returns:
            tuple: (weighted_features, log_time_pred)
                - weighted_features: Aggregated bag features [1, D]
                - log_time_pred: Predicted log survival time [1, 1]
        """
        # Ensure proper attention weight dimensions
        attention_weights = attention_weights.view(-1, 1)
        
        # Compute weighted features and aggregate
        weighted_features = features * attention_weights
        bag_features = weighted_features.sum(dim=0, keepdim=True)
        
        # Predict log survival time
        log_time_pred = self.time_predictor(bag_features)
        
        return bag_features, log_time_pred

class tapfm_multitask_survival_aggregator(nn.Module):
    """
    Multi-task logistic regression aggregator for discretized survival prediction
    """
    def __init__(self, ndim, time_intervals, dropout_rate=0.5):
        super().__init__()
        self.ndim = ndim
        self.time_intervals = time_intervals
        self.num_intervals = len(time_intervals)
        
        # Shared feature processing
        self.shared_layers = nn.Sequential(
            nn.LayerNorm(ndim),
            nn.Linear(ndim, ndim // 2),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(ndim // 2, ndim // 4),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        # Separate logistic regression heads for each time interval
        self.interval_classifiers = nn.ModuleList([
            nn.Linear(ndim // 4, 1) for _ in range(self.num_intervals)
        ])
   
    def forward(self, features, attention_weights):
        """
        Forward pass computing survival logits for each time interval
        
        Args:
            features: Feature vectors [N, D] 
            attention_weights: Attention weights [N] for each patch
        Returns:
            tuple: (weighted_features, interval_logits)
                - weighted_features: Aggregated bag features [1, D]
                - interval_logits: Raw logits for each interval [1, num_intervals]
        """
        attention_weights = attention_weights.view(-1, 1)
        
        # Compute weighted features and aggregate
        weighted_features = features * attention_weights
        bag_features = weighted_features.sum(dim=0, keepdim=True)
        
        # Shared feature processing
        shared_repr = self.shared_layers(bag_features)
        
        # Predict survival logits for each time interval (no sigmoid here)
        interval_logits = []
        for classifier in self.interval_classifiers:
            logit = classifier(shared_repr)
            interval_logits.append(logit)
        
        interval_logits = torch.cat(interval_logits, dim=1)
        
        return bag_features, interval_logits

def create_time_intervals(survival_times, num_intervals=10):
    """
    Create time intervals for discretization based on data distribution
    
    Args:
        survival_times: Array of survival times from training data
        num_intervals: Number of intervals to create
    
    Returns:
        time_intervals: List of time thresholds defining intervals
    """
    import numpy as np
    
    # Use quantiles to create intervals that ensure reasonable sample distribution
    quantiles = np.linspace(0, 1, num_intervals + 1)
    time_intervals = np.quantile(survival_times, quantiles[1:])  # Exclude 0th quantile
    
    return time_intervals.tolist()

class MultiTaskSurvivalLoss(nn.Module):
    """
    Multi-task logistic regression loss for survival analysis
    """
    def __init__(self, time_intervals, eps=1e-8):
        super().__init__()
        self.time_intervals = time_intervals
        self.num_intervals = len(time_intervals)
        self.eps = eps
        self.bce_with_logits_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, interval_logits, time, event):
        """
        Calculate multi-task survival loss for individual slide
        
        Args:
            interval_logits: Predicted survival logits [1, num_intervals]
            time: Actual survival time for current slide
            event: Event indicator (1=event, 0=censored)
        
        Returns:
            loss: Multi-task survival loss value
            task_losses: Individual losses for each time interval
        """
        interval_logits = interval_logits.squeeze()
        
        # Create target labels for each time interval
        targets = torch.zeros(self.num_intervals, device=interval_logits.device)
        weights = torch.ones(self.num_intervals, device=interval_logits.device)
        
        for i, threshold in enumerate(self.time_intervals):
            if event == 1:  # Uncensored event
                # Patient survives past this interval if event time > threshold
                targets[i] = 1.0 if time > threshold else 0.0
                weights[i] = 1.0  # Full weight for uncensored observations
            else:  # Censored observation
                if time > threshold:
                    targets[i] = 1.0  # Definitely survived past this interval
                    weights[i] = 1.0  # Full weight for known survival
                else:
                    targets[i] = 0.0  # Unknown survival status at this interval
                    weights[i] = 0.3  # Reduced weight for uncertain cases
        
        # Compute binary cross entropy with logits for each interval
        task_losses = self.bce_with_logits_loss(interval_logits, targets)
        
        # Apply uncertainty weighting for censored observations
        weighted_task_losses = task_losses * weights
        
        # Apply temporal weighting (later intervals weighted more heavily)
        temporal_weights = torch.linspace(0.5, 1.5, self.num_intervals, device=interval_logits.device)
        final_weighted_losses = weighted_task_losses * temporal_weights
        
        total_loss = final_weighted_losses.mean()
        
        return total_loss, task_losses
    
    def predict_risk_score(self, interval_logits):
        """
        Convert interval logits to single risk score for concordance evaluation
        
        Args:
            interval_logits: Predicted survival logits [1, num_intervals]
        
        Returns:
            risk_score: Single risk score derived from interval predictions
        """
        # Convert logits to probabilities
        interval_probs = torch.sigmoid(interval_logits.squeeze())
        
        # Calculate survival function and compute expected survival time
        survival_probs = torch.cumprod(interval_probs, dim=0)
        
        # Convert to risk score (higher risk = lower expected survival)
        expected_survival = torch.sum(survival_probs * torch.tensor(self.time_intervals, device=interval_logits.device))
        risk_score = 1.0 / (expected_survival + self.eps)
        
        return risk_score

class MiniBatchGammaMILFocalLoss(nn.Module):
    """
    Balanced Focal Loss for Multiple Instance Learning with adaptive gamma values per class.
    Updates gamma values after processing a mini-batch of slides (default: 50 slides).
    """
    def __init__(self, class_weights=None, alpha=None, base_gamma=2.0, eps=1e-6, 
                 global_balancing=False, modulating_factor=1.0, 
                 gamma_min=0.5, gamma_max=5.0, gamma_adaptation_rate=0.01,
                 mini_batch_size=50):
        """
        Args:
            base_gamma: Initial focusing parameter for all classes
            alpha: Weighting factor for positive/negative balance
            eps: Small constant for numerical stability
            class_weights: Fixed weights for each class based on dataset statistics
            global_balancing: Whether to apply additional global loss balancing
            modulating_factor: Scaling factor for the modulating term
            gamma_min: Minimum value for adaptive gamma
            gamma_max: Maximum value for adaptive gamma
            gamma_adaptation_rate: Rate at which gamma values are updated
            mini_batch_size: Number of slides to process before updating gamma values
        """
        super().__init__()
        self.base_gamma = base_gamma
        self.eps = eps
        self.class_weights = class_weights
        self.global_balancing = global_balancing
        self.modulating_factor = modulating_factor
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_adaptation_rate = gamma_adaptation_rate
        self.mini_batch_size = mini_batch_size
        
        # For tracking per-class loss history (used to adjust gamma)
        self.register_buffer('running_class_losses', None)
        
        # For tracking per-class gamma values
        self.register_buffer('class_gamma_values', None)
        
        # For mini-batch tracking
        self.register_buffer('accumulated_class_losses', None)
        self.register_buffer('slides_processed', torch.tensor(0))
        
        # EMA smoothing factor for loss tracking
        self.smoothing = 0.95

        # Handle alpha - either single value or class-specific values
        if alpha is None:
            self.alpha = 0.25  # Default value if none provided
            self.class_specific_alpha = False
        elif isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
            self.class_specific_alpha = False
        else:
            # Convert to tensor if list/numpy array
            if isinstance(alpha, (list, tuple, np.ndarray)):
                self.alpha = torch.tensor(alpha, dtype=torch.float32)
            else:
                self.alpha = alpha
            self.class_specific_alpha = True
    
    def _initialize_tracking_buffers(self, num_classes, device):
        """Initialize buffers for tracking losses and gamma values if not already done"""
        if self.running_class_losses is None:
            self.register_buffer('running_class_losses', 
                                torch.ones(num_classes, device=device))
        
        if self.class_gamma_values is None:
            self.register_buffer('class_gamma_values', 
                                torch.ones(num_classes, device=device) * self.base_gamma)
        
        if self.accumulated_class_losses is None:
            self.register_buffer('accumulated_class_losses',
                               torch.zeros(num_classes, device=device))
        
    def _update_gamma_values(self, accumulated_losses):
        """
        Update gamma values based on accumulated loss from mini-batch of slides
        
        Logic:
        - If a class consistently shows high loss, reduce its gamma to prevent overfitting
        - If a class shows low loss, increase gamma to maintain focus on hard examples
        """
        # Calculate average losses for this mini-batch
        avg_losses = accumulated_losses / self.mini_batch_size
        
        # Update running average of losses
        if self.running_class_losses is None:
            self.running_class_losses = avg_losses.detach()
        else:
            self.running_class_losses = self.smoothing * self.running_class_losses + \
                                       (1 - self.smoothing) * avg_losses.detach()
        
        # Calculate normalized loss values (between 0 and 1)
        max_loss = torch.max(self.running_class_losses)
        min_loss = torch.min(self.running_class_losses)
        epsilon = 1e-8  # Small value to prevent division by zero
        normalized_losses = (self.running_class_losses - min_loss) / (max_loss - min_loss + epsilon)
        
        # Higher normalized loss -> lower gamma (to prevent overfitting on difficult classes)
        # Lower normalized loss -> higher gamma (to maintain focus on hard examples for easy classes)
        target_gamma = self.gamma_max - normalized_losses * (self.gamma_max - self.gamma_min)
        
        # Update gamma values gradually using adaptation rate
        self.class_gamma_values = (1 - self.gamma_adaptation_rate) * self.class_gamma_values + \
                                 self.gamma_adaptation_rate * target_gamma
        
        # Ensure gamma values stay within bounds
        self.class_gamma_values = torch.clamp(self.class_gamma_values, 
                                             min=self.gamma_min, 
                                             max=self.gamma_max)
        
        # Reset accumulated losses and slides counter for next mini-batch
        self.accumulated_class_losses.zero_()
        self.slides_processed.zero_()
        
        return self.class_gamma_values
        
    def forward(self, slide_prediction, slide_label):
        """
        Calculate loss on slide-level predictions with adaptive gamma per class.
        
        Args:
            slide_prediction: Model prediction for the whole slide, shape [1, num_classes]
            slide_label: Ground truth label for the whole slide, shape [1, num_classes]
            
        Returns:
            total_loss: Scalar loss value
            class_losses: Per-class loss values for monitoring
            current_gammas: Current gamma values for each class (if updated this step)
        """
        # Handle potential batch dimension (though in MIL this is typically 1)
        if len(slide_prediction.shape) > 1 and slide_prediction.shape[0] > 1:
            # If somehow we have a batch of slides, take the mean across them
            return self._compute_loss(slide_prediction, slide_label)
        
        # Ensure we have the right shapes
        if len(slide_prediction.shape) == 1:
            slide_prediction = slide_prediction.unsqueeze(0)
        if len(slide_label.shape) == 1:
            slide_label = slide_label.unsqueeze(0)
        
        return self._compute_loss(slide_prediction, slide_label)
    
    def _compute_loss(self, prediction, target):
        # Apply label smoothing
        target = smooth_labels(target, epsilon=0.1)

        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(prediction)
        num_classes = prediction.shape[1]
        
        # Ensure we have tracking buffers initialized
        self._initialize_tracking_buffers(num_classes, prediction.device)

        # Ensure alpha has correct format if class-specific
        if self.class_specific_alpha:
            if len(self.alpha) != num_classes:
                raise ValueError(f"Expected alpha to have {num_classes} elements, got {len(self.alpha)}")
            
            # Move alpha to the correct device
            if isinstance(self.alpha, torch.Tensor) and self.alpha.device != prediction.device:
                self.alpha = self.alpha.to(prediction.device)

        # Initialize tensor to hold per-class losses
        class_losses = torch.zeros(num_classes, device=prediction.device)
        
        # Process each class separately with its own adaptive gamma
        for c in range(num_classes):
            # Get probabilities and targets for this class
            class_probs = probs[:, c]
            class_targets = target[:, c]

            # Get class-specific alpha if available, otherwise use global alpha
            if self.class_specific_alpha:
                if isinstance(self.alpha, torch.Tensor):
                    class_alpha = self.alpha[c].item()
                else:
                    class_alpha = self.alpha[c]
            else:
                class_alpha = self.alpha

            # Get class-specific gamma value
            class_gamma = self.class_gamma_values[c]

            # Calculate pt (probability of the target class)
            pt = torch.where(class_targets > 0.5, class_probs, 1 - class_probs)
            
            # Calculate alpha weight based on target
            alpha_weight = torch.where(class_targets > 0.5, class_alpha, 1 - class_alpha)
            
            # Calculate modulating factor: (1 - p_t)^gamma using class-specific gamma
            modulating_factor = self.modulating_factor * (1.0 - pt) ** class_gamma
            
            # Calculate binary cross entropy
            bce = -torch.log(pt + self.eps)
            
            # Combine components: alpha_t * (1-pt)^gamma * BCE
            loss = alpha_weight * modulating_factor * bce
            
            # Take mean for this class
            class_losses[c] = loss.mean()
            # print(f"For class {c}: loss = {loss.mean()}, alpha = {alpha_weight}, modulating factor = {modulating_factor}, bce = {bce}, probability = {class_probs}, target = {class_targets} ")
        # Accumulate losses for mini-batch update
        self.accumulated_class_losses += class_losses.detach()
        self.slides_processed += 1
        
        # Check if we should update gamma values
        updated_gammas = None
        if self.slides_processed >= self.mini_batch_size:
            updated_gammas = self._update_gamma_values(self.accumulated_class_losses)
        
        # Apply class weights if provided
        if self.class_weights is not None:
            if isinstance(self.class_weights, (list, tuple, np.ndarray)):
                weights = torch.tensor(self.class_weights, device=class_losses.device, dtype=torch.float32)
            else:
                weights = self.class_weights.to(class_losses.device)
            
            class_losses = class_losses * weights
        # print(f"Losses after class weights: {class_losses}, class weights = {weights}")
        # Apply global loss balancing if enabled
        if self.global_balancing and hasattr(self, '_apply_global_balancing'):
            class_losses = self._apply_global_balancing(class_losses)
        # print(f"Losses after global smoothing: {class_losses}")
        # Return the summed loss, individual class losses, and current gamma values if updated
        return class_losses.sum(), class_losses, updated_gammas
    
    def _apply_global_balancing(self, class_losses):
        """Apply global balancing to class losses"""
        # Initialize running_losses on first call
        if not hasattr(self, 'running_losses') or self.running_losses is None:
            self.register_buffer('running_losses', class_losses.detach())
        else:
            # Update running average of losses
            self.running_losses = self.smoothing * self.running_losses + \
                                 (1 - self.smoothing) * class_losses.detach()
        
        # Compute relative loss magnitudes
        normalized_losses = class_losses / (self.running_losses + self.eps)
        
        # Balance the class losses
        return normalized_losses

    def get_current_gamma_values(self):
        """Return current gamma values for logging/monitoring"""
        if self.class_gamma_values is not None:
            return self.class_gamma_values.detach().cpu().numpy()
        return None
        
    def get_mini_batch_progress(self):
        """Return current mini-batch progress (slides processed / mini_batch_size)"""
        if self.slides_processed is not None:
            return self.slides_processed.item(), self.mini_batch_size
        return 0, self.mini_batch_size

def smooth_labels(target, epsilon=0.1):
    """Apply label smoothing to binary targets
    For multi-label: 1 becomes 1-epsilon, 0 becomes epsilon
    """
    return target * (1 - epsilon) + epsilon * 0.5 

def save_batch_progress(epoch, batch_file_idx, args):
    """Save only the batch index for resuming mid-epoch"""
    
    progress = {
        'epoch': epoch,
        'batch_file_idx': batch_file_idx
    }
    
    progress_path = os.path.join(args.outdir, 'batch_progress.json')
    with open(progress_path, 'w') as f:
        json.dump(progress, f)
    
    print(f"Saved progress: epoch {epoch}, batch {batch_file_idx}")