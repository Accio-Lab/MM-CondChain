"""
Data loading utilities for MM-CondChain evaluation.
"""

import os
import json
import glob
from typing import List, Dict, Any, Optional, Iterator


class MMCondChainDataset:
    """
    Dataset loader for MM-CondChain benchmark.
    
    Supports loading from:
    - HuggingFace dataset (Accio-Lab/MM-CondChain)
    - Local JSONL file
    """
    
    def __init__(
        self,
        data_path: str,
        image_root: str,
        domain: Optional[str] = None,
        use_hf: bool = False,
    ):
        """
        Initialize dataset loader.
        
        Args:
            data_path: Path to JSONL file or HuggingFace dataset name
            image_root: Root directory for images
            domain: Domain filter (natural, chart, gui). If None, load all.
            use_hf: Whether to load from HuggingFace Hub
        """
        self.data_path = data_path
        self.image_root = image_root
        self.domain = domain
        self.use_hf = use_hf
        self._data: List[Dict[str, Any]] = []
        self._load_data()
    
    def _load_data(self):
        """Load data from source."""
        if self.use_hf:
            self._load_from_hf()
        else:
            self._load_from_jsonl()
    
    def _load_from_hf(self):
        """Load data from HuggingFace Hub."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Please install datasets: pip install datasets")
        
        if self.domain:
            ds = load_dataset(self.data_path, split=self.domain)
            self._data = list(ds)
        else:
            ds = load_dataset(self.data_path)
            for split_name in ds.keys():
                self._data.extend(list(ds[split_name]))
    
    def _load_from_jsonl(self):
        """Load data from local JSONL or JSON file."""
        with open(self.data_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.split("\n") if line.strip()]
        
        for idx, item in enumerate(data):
            if self.domain is None or item.get("domain") == self.domain:
                if "id" not in item:
                    seed = item.get("seed")
                    item["id"] = f"{idx}" if seed is None else f"{idx}_seed{seed}"
                self._data.append(item)
    
    def resolve_single_image(self, relative_path: str) -> str:
        """
        Resolve single image path for Natural/Chart domains.
        
        Args:
            relative_path: Relative path from JSONL (e.g., "images/natural/xxx.jpg")
            
        Returns:
            Absolute path to the image
        """
        return os.path.join(self.image_root, relative_path)
    
    def resolve_gui_images(self, trajectory_folder: str) -> List[str]:
        """
        Resolve GUI trajectory images.
        
        Args:
            trajectory_folder: Relative path to trajectory folder
            
        Returns:
            Sorted list of absolute paths to all PNG images in the folder
        """
        folder_path = os.path.join(self.image_root, trajectory_folder)
        
        patterns = ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
        image_files = []
        for pattern in patterns:
            image_files.extend(glob.glob(os.path.join(folder_path, pattern)))
        
        image_files = sorted(set(image_files))
        return image_files
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for item in self._data:
            yield self._resolve_item(item)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._resolve_item(self._data[idx])
    
    def _resolve_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve image paths for an item.
        
        Returns item with added:
        - image_path: str (for Natural/Chart)
        - image_paths: List[str] (for GUI)
        """
        resolved = dict(item)
        domain = item.get("domain", "natural")
        image_rel = item.get("image") or item.get("image_path", "")
        
        if domain == "gui":
            resolved["image_paths"] = self.resolve_gui_images(image_rel)
            resolved["image_path"] = None
        else:
            if os.path.isabs(image_rel) and os.path.exists(image_rel):
                resolved["image_path"] = image_rel
            else:
                resolved["image_path"] = self.resolve_single_image(image_rel)
            resolved["image_paths"] = None
        
        return resolved


def load_dataset_for_eval(
    data_path: str,
    image_root: str,
    domain: Optional[str] = None,
    use_hf: bool = False,
) -> MMCondChainDataset:
    """
    Convenience function to load MM-CondChain dataset.
    
    Args:
        data_path: Path to JSONL file or HuggingFace dataset name
        image_root: Root directory for images
        domain: Domain filter (natural, chart, gui)
        use_hf: Whether to load from HuggingFace Hub
        
    Returns:
        MMCondChainDataset instance
    """
    return MMCondChainDataset(
        data_path=data_path,
        image_root=image_root,
        domain=domain,
        use_hf=use_hf,
    )
