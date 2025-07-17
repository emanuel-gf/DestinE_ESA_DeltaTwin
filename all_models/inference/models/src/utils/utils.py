import yaml
from loguru import logger
import os
import numpy as np
import torch
from pathlib import Path

def load_config(config_path: str = "cfg/config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration YAML file.

    Returns:
        dict: The loaded configuration dictionary.
    """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.success(f"Loaded configuration from {config_path}")
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file {config_path} not found")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse YAML configuration: {e}")
        return {}
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return {}


def load_numpy_as_tensor(file_path: str, device: torch.device = None, 
                        add_batch_dim: bool = True, permute_to_chw: bool = True) -> torch.Tensor:
    """
    Load a NumPy array file and convert it back to PyTorch tensor with proper formatting.
    
    Args:
        file_path: Path to the NumPy file
        device: Device to load the tensor on (CPU/CUDA)
        add_batch_dim: Whether to add batch dimension (unsqueeze(0))
        permute_to_chw: Whether to permute from HWC to CHW format
    
    Returns:
        PyTorch tensor with shape [B, C, H, W] if add_batch_dim=True, else [C, H, W]
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npz':
        try:
        # Load from compressed .npz
            with np.load(file_path) as data:
                numpy_array = data['array']
        except:
            with np.load(file_path) as data:
                numpy_array = data['arr_0']
    elif file_path.suffix == '.npy':
        # Load from regular .npy
        numpy_array = np.load(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # Convert to PyTorch tensor and ensure float32
    tensor = torch.from_numpy(numpy_array).float()

    ## check shape
    if not (tensor.shape[0]==1) & (tensor.shape[1]==3) & (tensor.shape[2]==1024):
        # Apply transformations to match the expected format
        if permute_to_chw and tensor.dim() == 3:
            # Permute from HWC to CHW: [H, W, C] -> [C, H, W]
            tensor = tensor.permute(2, 0, 1)

        
        if (add_batch_dim==True) & (tensor.dim()==3):
            # Add batch dimension: [C, H, W] -> [B, C, H, W]
            tensor = tensor.unsqueeze(0)
    
    if device is not None:
        tensor = tensor.to(device)
    
    logger.info(f"NumPy array loaded as tensor: {file_path}")
    logger.info(f"Final tensor shape: {tensor.shape}")
    return tensor


def save_tensor_as_numpy(tensor: torch.Tensor, file_path: str, compress: bool = True) -> None:
    """
    Save a PyTorch tensor as a NumPy array file.
    
    Args:
        tensor: PyTorch tensor to save
        file_path: Path where to save the file
        compress: Whether to use compressed format (.npz) or not (.npy)
    """
    # Convert tensor to numpy (move to CPU first if on GPU)
    if tensor.is_cuda:
        numpy_array = tensor.cpu().numpy()
    else:
        numpy_array = tensor.numpy()
    
    file_path = Path(file_path)
    
    if compress:
        # Save as compressed .npz
        if not file_path.suffix:
            file_path = file_path.with_suffix('.npz')
        np.savez_compressed(file_path, array=numpy_array)
    else:
        # Save as regular .npy
        if not file_path.suffix:
            file_path = file_path.with_suffix('.npy')
        np.save(file_path, numpy_array)
    
    logger.info(f"Tensor saved as NumPy array: {file_path}")
