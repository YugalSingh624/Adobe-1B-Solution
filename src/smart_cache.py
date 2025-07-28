#!/usr/bin/env python3
"""
Smart caching system for improved performance and reduced redundant processing.
"""

import json
import hashlib
import pickle
import os
from typing import Any, Dict, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SmartCacheManager:
    """
    Advanced caching system for embeddings, computations, and results.
    """
    
    def __init__(self, cache_dir: str = "cache", max_cache_size_mb: int = 100):
        self.cache_dir = cache_dir
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size_bytes': 0
        }
        
        # Create cache directories
        self.embedding_cache_dir = os.path.join(cache_dir, "embeddings")
        self.computation_cache_dir = os.path.join(cache_dir, "computations")
        self.result_cache_dir = os.path.join(cache_dir, "results")
        
        for dir_path in [self.embedding_cache_dir, self.computation_cache_dir, self.result_cache_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def get_cache_key(self, data: Any) -> str:
        """Generate a unique cache key for any data."""
        if isinstance(data, str):
            content = data
        elif isinstance(data, dict):
            content = json.dumps(data, sort_keys=True)
        elif isinstance(data, (list, tuple)):
            content = str(data)
        else:
            content = str(data)
            
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def cache_embedding(self, key: str, embedding: np.ndarray, metadata: Dict = None) -> bool:
        """Cache an embedding with metadata."""
        try:
            cache_file = os.path.join(self.embedding_cache_dir, f"{key}.pkl")
            cache_data = {
                'embedding': embedding,
                'metadata': metadata or {},
                'timestamp': datetime.now().isoformat(),
                'type': 'embedding'
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
                
            self._update_cache_stats(cache_file)
            logger.debug(f"Cached embedding: {key}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache embedding {key}: {e}")
            return False
    
    def get_cached_embedding(self, key: str, max_age_hours: int = 24) -> Optional[Tuple[np.ndarray, Dict]]:
        """Retrieve cached embedding if it exists and is fresh."""
        try:
            cache_file = os.path.join(self.embedding_cache_dir, f"{key}.pkl")
            
            if not os.path.exists(cache_file):
                self.cache_stats['misses'] += 1
                return None
                
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                
            # Check age
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                os.remove(cache_file)
                self.cache_stats['misses'] += 1
                return None
                
            self.cache_stats['hits'] += 1
            logger.debug(f"Cache hit for embedding: {key}")
            return cache_data['embedding'], cache_data['metadata']
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached embedding {key}: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    def cache_computation_result(self, key: str, result: Any, computation_type: str) -> bool:
        """Cache computation results."""
        try:
            cache_file = os.path.join(self.computation_cache_dir, f"{key}.json")
            cache_data = {
                'result': result,
                'computation_type': computation_type,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, default=str)
                
            self._update_cache_stats(cache_file)
            logger.debug(f"Cached computation result: {key}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache computation {key}: {e}")
            return False
    
    def get_cached_computation(self, key: str, max_age_hours: int = 6) -> Optional[Any]:
        """Retrieve cached computation result."""
        try:
            cache_file = os.path.join(self.computation_cache_dir, f"{key}.json")
            
            if not os.path.exists(cache_file):
                self.cache_stats['misses'] += 1
                return None
                
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # Check age
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                os.remove(cache_file)
                self.cache_stats['misses'] += 1
                return None
                
            self.cache_stats['hits'] += 1
            logger.debug(f"Cache hit for computation: {key}")
            return cache_data['result']
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached computation {key}: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    def cache_pipeline_result(self, persona: str, job: str, result: Dict) -> bool:
        """Cache complete pipeline results."""
        try:
            key = self.get_cache_key(f"{persona}_{job}")
            cache_file = os.path.join(self.result_cache_dir, f"pipeline_{key}.json")
            
            cache_data = {
                'persona': persona,
                'job': job,
                'result': result,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)
                
            self._update_cache_stats(cache_file)
            logger.debug(f"Cached pipeline result for: {persona[:30]}...")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to cache pipeline result: {e}")
            return False
    
    def get_cached_pipeline_result(self, persona: str, job: str, max_age_hours: int = 2) -> Optional[Dict]:
        """Retrieve cached pipeline result."""
        try:
            key = self.get_cache_key(f"{persona}_{job}")
            cache_file = os.path.join(self.result_cache_dir, f"pipeline_{key}.json")
            
            if not os.path.exists(cache_file):
                self.cache_stats['misses'] += 1
                return None
                
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # Check age (pipeline results should be fresh)
            cached_time = datetime.fromisoformat(cache_data['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=max_age_hours):
                os.remove(cache_file)
                self.cache_stats['misses'] += 1
                return None
                
            self.cache_stats['hits'] += 1
            logger.info(f"Cache hit for pipeline: {persona[:30]}...")
            return cache_data['result']
            
        except Exception as e:
            logger.warning(f"Failed to retrieve cached pipeline result: {e}")
            self.cache_stats['misses'] += 1
            return None
    
    def _update_cache_stats(self, cache_file: str):
        """Update cache statistics."""
        try:
            file_size = os.path.getsize(cache_file)
            self.cache_stats['size_bytes'] += file_size
            
            # Check if cache is too large
            if self.cache_stats['size_bytes'] > self.max_cache_size_bytes:
                self._evict_old_cache_entries()
                
        except Exception as e:
            logger.warning(f"Failed to update cache stats: {e}")
    
    def _evict_old_cache_entries(self):
        """Evict old cache entries to free space."""
        try:
            all_cache_files = []
            
            # Collect all cache files with timestamps
            for cache_dir in [self.embedding_cache_dir, self.computation_cache_dir, self.result_cache_dir]:
                for filename in os.listdir(cache_dir):
                    filepath = os.path.join(cache_dir, filename)
                    if os.path.isfile(filepath):
                        mtime = os.path.getmtime(filepath)
                        all_cache_files.append((filepath, mtime))
            
            # Sort by modification time (oldest first)
            all_cache_files.sort(key=lambda x: x[1])
            
            # Remove oldest files until under limit
            bytes_freed = 0
            for filepath, _ in all_cache_files:
                if self.cache_stats['size_bytes'] - bytes_freed <= self.max_cache_size_bytes * 0.8:
                    break
                    
                try:
                    file_size = os.path.getsize(filepath)
                    os.remove(filepath)
                    bytes_freed += file_size
                    self.cache_stats['evictions'] += 1
                    logger.debug(f"Evicted cache file: {os.path.basename(filepath)}")
                except Exception as e:
                    logger.warning(f"Failed to evict cache file {filepath}: {e}")
            
            self.cache_stats['size_bytes'] -= bytes_freed
            logger.info(f"Cache cleanup: freed {bytes_freed} bytes, evicted {self.cache_stats['evictions']} files")
            
        except Exception as e:
            logger.error(f"Cache eviction failed: {e}")
    
    def get_cache_statistics(self) -> Dict:
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hit_rate_percent': round(hit_rate, 2),
            'total_hits': self.cache_stats['hits'],
            'total_misses': self.cache_stats['misses'],
            'total_evictions': self.cache_stats['evictions'],
            'cache_size_mb': round(self.cache_stats['size_bytes'] / (1024 * 1024), 2),
            'max_cache_size_mb': round(self.max_cache_size_bytes / (1024 * 1024), 2)
        }
    
    def clear_cache(self, cache_type: str = 'all'):
        """Clear specific or all cache types."""
        try:
            if cache_type in ['all', 'embeddings']:
                self._clear_directory(self.embedding_cache_dir)
            if cache_type in ['all', 'computations']:
                self._clear_directory(self.computation_cache_dir)
            if cache_type in ['all', 'results']:
                self._clear_directory(self.result_cache_dir)
                
            if cache_type == 'all':
                self.cache_stats = {'hits': 0, 'misses': 0, 'evictions': 0, 'size_bytes': 0}
                
            logger.info(f"Cleared {cache_type} cache")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def _clear_directory(self, directory: str):
        """Clear all files in a directory."""
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                os.remove(filepath)

# Global cache manager instance
smart_cache = SmartCacheManager()
