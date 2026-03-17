"""
API-based evaluator for MM-CondChain.

Supports:
- OpenAI API (GPT-4o, GPT-4V, etc.)
- Azure OpenAI API
- vLLM with OpenAI-compatible API
"""

import os
from typing import Optional, List

from openai import OpenAI, AzureOpenAI

from .base import BaseEvaluator
from ..utils import build_message_content, parse_answer


class APIEvaluator(BaseEvaluator):
    """
    Evaluator using OpenAI-compatible APIs.
    
    Supports OpenAI, Azure OpenAI, and vLLM backends.
    """
    
    def __init__(
        self,
        model: str,
        api_type: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
        stream: bool = False,
    ):
        """
        Initialize API evaluator.
        
        Args:
            model: Model name (e.g., gpt-4o, Qwen/Qwen2.5-VL-7B)
            api_type: One of "openai", "azure", "vllm"
            api_key: API key (defaults to env var)
            base_url: Base URL for vLLM server
            azure_endpoint: Azure OpenAI endpoint
            api_version: Azure API version
            stream: Whether to use streaming
        """
        self.model = model
        self.api_type = api_type
        self.stream = stream
        
        self.client = self._create_client(
            api_type=api_type,
            api_key=api_key,
            base_url=base_url,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )
    
    def _create_client(
        self,
        api_type: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        api_version: Optional[str] = None,
    ) -> OpenAI:
        """Create API client based on api_type."""
        if api_type == "openai":
            return OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        
        elif api_type == "azure":
            return AzureOpenAI(
                api_key=api_key or os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=azure_endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=api_version or "2024-02-01",
            )
        
        elif api_type == "vllm":
            if not base_url:
                raise ValueError("base_url is required for vLLM API")
            return OpenAI(
                api_key=api_key or "EMPTY",
                base_url=base_url,
            )
        
        else:
            raise ValueError(f"Unknown api_type: {api_type}")
    
    @property
    def supports_parallel(self) -> bool:
        """API evaluators support parallel execution."""
        return True
    
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
            image_path: Single image path (Natural/Chart)
            image_paths: List of image paths (GUI)
            
        Returns:
            Parsed answer string
        """
        content = build_message_content(instruction, image_path, image_paths)
        
        if content is None:
            return "None"
        
        messages = [{"role": "user", "content": content}]
        
        try:
            if self.stream:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    stream=True,
                )
                text = ""
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        text += chunk.choices[0].delta.content
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )
                text = response.choices[0].message.content or ""
            
            return parse_answer(text)
        
        except Exception as e:
            print(f"Error during inference: {e}")
            return "None"


def create_evaluator(
    model: str,
    api_type: str = "openai",
    **kwargs,
) -> APIEvaluator:
    """
    Factory function to create an API evaluator.
    
    Args:
        model: Model name
        api_type: One of "openai", "azure", "vllm"
        **kwargs: Additional arguments passed to APIEvaluator
        
    Returns:
        APIEvaluator instance
    """
    return APIEvaluator(model=model, api_type=api_type, **kwargs)
