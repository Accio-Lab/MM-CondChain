"""
Base evaluator interface for MM-CondChain.
"""

from abc import ABC, abstractmethod
from typing import Optional, List


class BaseEvaluator(ABC):
    """
    Abstract base class for model evaluators.
    
    All evaluators must implement the get_answer method.
    """
    
    @property
    def supports_parallel(self) -> bool:
        """
        Whether this evaluator supports parallel execution.
        
        API-based evaluators typically support parallel (thread-safe).
        Local model evaluators may not (GPU resource conflicts).
        """
        return True
    
    @abstractmethod
    def get_answer(
        self,
        instruction: str,
        image_path: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
    ) -> str:
        """
        Get model answer for an instruction with image(s).
        
        Args:
            instruction: The instruction text
            image_path: Single image path (for Natural/Chart domains)
            image_paths: List of image paths (for GUI multi-frame)
            
        Returns:
            Parsed answer string (e.g., "A1", "B2", "None")
        """
        raise NotImplementedError
    
    def get_answers_batch(
        self,
        items: List[dict],
    ) -> List[str]:
        """
        Batch inference for efficiency (optional).
        
        Default implementation calls get_answer sequentially.
        Subclasses can override for true batch processing.
        
        Args:
            items: List of dicts with 'instruction', 'image_path', 'image_paths'
            
        Returns:
            List of parsed answers
        """
        results = []
        for item in items:
            ans = self.get_answer(
                instruction=item.get("instruction", ""),
                image_path=item.get("image_path"),
                image_paths=item.get("image_paths"),
            )
            results.append(ans)
        return results
