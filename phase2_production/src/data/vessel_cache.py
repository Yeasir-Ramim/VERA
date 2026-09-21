"""
VERA Phase 2 - Vessel Segmentation Caching

Utilities for pre-computing and caching vessel segmentation maps
to accelerate training iterations.
"""

import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict
from tqdm import tqdm
import logging
import hashlib

logger = logging.getLogger(__name__)


class VesselCache:
    """
    Manager for caching pre-computed vessel segmentation maps.
    
    Stores vessel probability maps to disk to avoid recomputing
    during every training epoch.
    """
    
    def __init__(
        self,
        cache_dir: str,
        image_size: int = 512,
        cache_format: str = 'png',
        use_compression: bool = True
    ):
        """
        Initialize vessel cache manager.
        
        Args:
            cache_dir: Directory to store cached vessel maps
            image_size: Size of cached images
            cache_format: Format for saving (png, npy, pt)
            use_compression: Use compression for storage
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.image_size = image_size
        self.cache_format = cache_format
        self.use_compression = use_compression
        
        # Cache metadata
        self.metadata_path = self.cache_dir / "cache_metadata.txt"
        self.metadata = self._load_metadata()
        
        logger.info(f"Initialized VesselCache at {self.cache_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata if exists."""
        metadata = {}
        
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        key, value = line.strip().split(':', 1)
                        metadata[key] = value
        
        return metadata
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_path, 'w') as f:
            for key, value in self.metadata.items():
                f.write(f"{key}:{value}\n")
    
    def _get_cache_key(self, image_id: str) -> str:
        """
        Generate unique cache key for image.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Cache key string
        """
        # Create hash of image_id for filename safety
        hash_obj = hashlib.md5(image_id.encode())
        cache_key = hash_obj.hexdigest()[:16]
        return cache_key
    
    def _get_cache_path(self, image_id: str) -> Path:
        """
        Get path for cached vessel map.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Path to cache file
        """
        cache_key = self._get_cache_key(image_id)
        
        if self.cache_format == 'png':
            filename = f"{cache_key}_vessel.png"
        elif self.cache_format == 'npy':
            filename = f"{cache_key}_vessel.npy"
        elif self.cache_format == 'pt':
            filename = f"{cache_key}_vessel.pt"
        else:
            filename = f"{cache_key}_vessel.bin"
        
        return self.cache_dir / filename
    
    def exists(self, image_id: str) -> bool:
        """
        Check if vessel map is cached.
        
        Args:
            image_id: Image identifier
            
        Returns:
            True if cached, False otherwise
        """
        cache_path = self._get_cache_path(image_id)
        return cache_path.exists()
    
    def save(self, image_id: str, vessel_map: np.ndarray):
        """
        Save vessel map to cache.
        
        Args:
            image_id: Image identifier
            vessel_map: Vessel probability map (H, W) or (H, W, 1)
        """
        cache_path = self._get_cache_path(image_id)
        
        # Ensure 2D
        if len(vessel_map.shape) == 3:
            vessel_map = vessel_map.squeeze()
        
        # Resize if needed
        if vessel_map.shape[0] != self.image_size or vessel_map.shape[1] != self.image_size:
            vessel_map = cv2.resize(vessel_map, (self.image_size, self.image_size),
                                   interpolation=cv2.INTER_LINEAR)
        
        # Save based on format
        if self.cache_format == 'png':
            # Convert to 8-bit for PNG
            vessel_map_8bit = (vessel_map * 255).astype(np.uint8)
            cv2.imwrite(str(cache_path), vessel_map_8bit)
        
        elif self.cache_format == 'npy':
            # Save as float32 numpy array
            if self.use_compression:
                np.savez_compressed(cache_path, vessel_map=vessel_map.astype(np.float32))
            else:
                np.save(cache_path, vessel_map.astype(np.float32))
        
        elif self.cache_format == 'pt':
            # Save as PyTorch tensor
            vessel_tensor = torch.from_numpy(vessel_map.astype(np.float32))
            torch.save(vessel_tensor, cache_path)
        
        else:
            raise ValueError(f"Unsupported cache format: {self.cache_format}")
        
        # Update metadata
        self.metadata[image_id] = str(cache_path)
    
    def load(self, image_id: str) -> Optional[np.ndarray]:
        """
        Load vessel map from cache.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Vessel probability map (H, W) or None if not cached
        """
        if not self.exists(image_id):
            return None
        
        cache_path = self._get_cache_path(image_id)
        
        # Load based on format
        if self.cache_format == 'png':
            vessel_map = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            vessel_map = vessel_map.astype(np.float32) / 255.0
        
        elif self.cache_format == 'npy':
            if cache_path.suffix == '.npz':
                data = np.load(cache_path)
                vessel_map = data['vessel_map']
            else:
                vessel_map = np.load(cache_path)
        
        elif self.cache_format == 'pt':
            vessel_tensor = torch.load(cache_path)
            vessel_map = vessel_tensor.numpy()
        
        else:
            raise ValueError(f"Unsupported cache format: {self.cache_format}")
        
        return vessel_map
    
    def clear(self):
        """Clear all cached vessel maps."""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata = {}
            logger.info(f"Cleared vessel cache at {self.cache_dir}")
    
    def get_cache_size(self) -> float:
        """
        Get total size of cache in MB.
        
        Returns:
            Cache size in megabytes
        """
        total_size = 0
        
        for file_path in self.cache_dir.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the cache.
        
        Returns:
            Dictionary with cache statistics
        """
        num_files = len(list(self.cache_dir.glob(f'*_vessel.{self.cache_format}')))
        cache_size_mb = self.get_cache_size()
        
        stats = {
            'num_cached': num_files,
            'cache_size_mb': cache_size_mb,
            'cache_dir': str(self.cache_dir),
            'cache_format': self.cache_format,
            'image_size': self.image_size
        }
        
        return stats


def batch_cache_vessel_maps(
    image_paths: list,
    vessel_segmenter,
    cache: VesselCache,
    batch_size: int = 8,
    device: str = 'cuda'
):
    """
    Pre-compute and cache vessel maps for a list of images.
    
    Args:
        image_paths: List of image file paths
        vessel_segmenter: Vessel segmentation model
        cache: VesselCache instance
        batch_size: Batch size for inference
        device: Device for computation
    """
    from src.data.preprocessing import VesselPreprocessor
    
    # Initialize preprocessor
    preprocessor = VesselPreprocessor(target_size=cache.image_size)
    
    # Filter images that need caching
    images_to_cache = []
    image_ids = []
    
    for image_path in image_paths:
        image_id = Path(image_path).stem
        if not cache.exists(image_id):
            images_to_cache.append(image_path)
            image_ids.append(image_id)
    
    if len(images_to_cache) == 0:
        logger.info("All vessel maps already cached")
        return
    
    logger.info(f"Caching vessel maps for {len(images_to_cache)} images...")
    
    # Set model to eval mode
    vessel_segmenter.eval()
    vessel_segmenter.to(device)
    
    # Process in batches
    num_batches = (len(images_to_cache) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Caching vessel maps"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(images_to_cache))
            
            batch_paths = images_to_cache[start_idx:end_idx]
            batch_ids = image_ids[start_idx:end_idx]
            
            # Load and preprocess batch
            batch_images = []
            for image_path in batch_paths:
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Failed to load image: {image_path}")
                    continue
                
                preprocessed = preprocessor.preprocess(image)
                batch_images.append(preprocessed)
            
            if len(batch_images) == 0:
                continue
            
            # Stack into batch tensor
            batch_tensor = np.stack(batch_images, axis=0)
            batch_tensor = torch.from_numpy(batch_tensor).permute(0, 3, 1, 2)  # NHWC -> NCHW
            batch_tensor = batch_tensor.to(device)
            
            # Run inference
            vessel_maps = vessel_segmenter(batch_tensor)
            
            # Apply sigmoid if needed
            if vessel_maps.shape[1] == 1:  # Single channel output
                vessel_maps = torch.sigmoid(vessel_maps)
            
            # Save to cache
            vessel_maps_np = vessel_maps.cpu().numpy()
            
            for idx, image_id in enumerate(batch_ids[:len(vessel_maps_np)]):
                vessel_map = vessel_maps_np[idx, 0]  # (H, W)
                cache.save(image_id, vessel_map)
    
    # Save metadata
    cache._save_metadata()
    
    # Print stats
    stats = cache.get_cache_stats()
    logger.info(f"Caching complete! Cached {stats['num_cached']} vessel maps, "
               f"cache size: {stats['cache_size_mb']:.2f} MB")


def create_vessel_cache(config: Dict) -> VesselCache:
    """
    Create vessel cache from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        VesselCache instance
    """
    data_config = config.get('data', {})
    vessel_seg_config = config.get('vessel_segmentation', {})
    
    cache = VesselCache(
        cache_dir=data_config.get('vessel_cache_dir', 'data/vessel_cache'),
        image_size=data_config.get('image_size', 512),
        cache_format=vessel_seg_config.get('cache_format', 'png'),
        use_compression=True
    )
    
    return cache
