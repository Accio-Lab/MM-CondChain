#!/usr/bin/env python3
"""
MM-CondChain Evaluation Script

Supports:
- OpenAI API (GPT-4o, GPT-4V, etc.)
- Azure OpenAI API
- vLLM with OpenAI-compatible API (open-source models)
"""

import os
import json
import argparse
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .data_loader import load_dataset_for_eval
from .metrics import compute_metrics, print_metrics
from .evaluator import create_evaluator, BaseEvaluator


def evaluate_sample(
    item: Dict[str, Any],
    evaluator: BaseEvaluator,
) -> Dict[str, Any]:
    """
    Evaluate a single sample (both true_path and false_path).
    
    Args:
        item: Dataset item with resolved image paths
        evaluator: Evaluator instance
        
    Returns:
        Result dict with predictions and ground truth
    """
    result = {
        "id": item.get("id", ""),
        "domain": item.get("domain", ""),
    }
    
    true_path = item.get("true_path", {})
    if true_path:
        true_pred = evaluator.get_answer(
            instruction=true_path.get("full_instruction"),
            image_path=item.get("image_path"),
            image_paths=item.get("image_paths"),
        )
        result["true_path_pred"] = true_pred
        result["true_path_gt"] = true_path.get("correct_answer")
    
    false_path = item.get("false_path", {})
    if false_path:
        false_pred = evaluator.get_answer(
            instruction=false_path.get("full_instruction"),
            image_path=item.get("image_path"),
            image_paths=item.get("image_paths"),
        )
        result["false_path_pred"] = false_pred
        result["false_path_gt"] = false_path.get("correct_answer")
    
    return result


def load_existing_results(output_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load existing results for resume functionality.
    
    Args:
        output_path: Path to output JSON file
        
    Returns:
        Dict mapping sample ID to result
    """
    if not os.path.exists(output_path):
        return {}
    
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        return {r["id"]: r for r in results}
    except Exception:
        return {}


def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def run_eval(
    evaluator: BaseEvaluator,
    dataset,
    output_path: str,
    workers: int = 8,
    resume: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    Run evaluation with the given evaluator.
    
    Args:
        evaluator: Evaluator instance
        dataset: MMCondChainDataset instance
        output_path: Path to save results
        workers: Number of parallel workers
        resume: Whether to resume from existing results
        
    Returns:
        Metrics dict
    """
    existing_results = {}
    if resume:
        existing_results = load_existing_results(output_path)
        print(f"Resuming from {len(existing_results)} existing results")
    
    items_to_eval = [item for item in dataset if item["id"] not in existing_results]
    print(f"Evaluating {len(items_to_eval)} samples...")
    
    results = list(existing_results.values())
    
    if evaluator.supports_parallel and workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate_sample, item, evaluator): item
                for item in items_to_eval
            }
            
            with tqdm(total=len(items_to_eval), desc="Evaluating") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    pbar.update(1)
                    
                    if len(results) % 50 == 0:
                        save_results(results, output_path)
    else:
        for item in tqdm(items_to_eval, desc="Evaluating"):
            result = evaluate_sample(item, evaluator)
            results.append(result)
            
            if len(results) % 50 == 0:
                save_results(results, output_path)
    
    save_results(results, output_path)
    print(f"Saved {len(results)} results to {output_path}")
    
    metrics = compute_metrics(results)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="MM-CondChain Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-infer data path from image_root (recommended)
  python -m eval.eval --api_type openai --model gpt-4o --domain natural --image_root /path/to/mm-condchain/images

  # Explicit data path
  python -m eval.eval --api_type openai --model gpt-4o --domain natural --data_path /path/to/natural.jsonl --image_root /path/to/images

  # Evaluate with vLLM (local server)
  python -m eval.eval --api_type vllm --base_url http://localhost:8000/v1 --model qwen3-vl-8b --domain gui --image_root /path/to/images

Directory structure (when using auto-infer):
  mm-condchain-data/
  ├── data/
  │   ├── natural.jsonl
  │   ├── chart.jsonl
  │   └── gui.jsonl
  └── images/
      ├── natural/
      ├── chart/
      └── gui/
        """,
    )
    
    parser.add_argument(
        "--api_type",
        type=str,
        choices=["openai", "azure", "vllm"],
        default="openai",
        help="API type to use",
    )
    parser.add_argument(
        "--base_url",
        type=str,
        default=None,
        help="Base URL for vLLM server (required for vllm)",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key (defaults to env var)",
    )
    parser.add_argument(
        "--azure_endpoint",
        type=str,
        default=None,
        help="Azure OpenAI endpoint",
    )
    parser.add_argument(
        "--api_version",
        type=str,
        default=None,
        help="Azure API version",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., gpt-4o, qwen3-vl-8b-instruct)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to JSONL/JSON file. If not provided, auto-infer from image_root/../data/{domain}.jsonl",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=".",
        help="Root directory for images (e.g., ./images or /path/to/mm-condchain-data/images)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        choices=["natural", "chart", "gui"],
        required=True,
        help="Domain to evaluate: natural, chart, or gui",
    )
    parser.add_argument(
        "--use_hf",
        action="store_true",
        help="Load from HuggingFace Hub",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON file path (default: ./results/{model}_{domain}.json)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Use streaming for inference (useful for vLLM)",
    )
    
    args = parser.parse_args()
    
    if args.data_path is None:
        data_dir = os.path.join(os.path.dirname(args.image_root.rstrip("/")), "data")
        args.data_path = os.path.join(data_dir, f"{args.domain}.jsonl")
        if not os.path.exists(args.data_path):
            json_path = os.path.join(data_dir, f"{args.domain}.json")
            if os.path.exists(json_path):
                args.data_path = json_path
            else:
                raise FileNotFoundError(f"Data file not found: {args.data_path} or {json_path}")
    
    if args.output is None:
        model_name = args.model.replace("/", "_")
        args.output = f"./results/{model_name}_{args.domain}.json"
    
    evaluator = create_evaluator(
        model=args.model,
        api_type=args.api_type,
        api_key=args.api_key,
        base_url=args.base_url,
        azure_endpoint=args.azure_endpoint,
        api_version=args.api_version,
        stream=args.stream,
    )
    
    print(f"Loading dataset from {args.data_path}...")
    dataset = load_dataset_for_eval(
        data_path=args.data_path,
        image_root=args.image_root,
        domain=args.domain,
        use_hf=args.use_hf,
    )
    print(f"Loaded {len(dataset)} samples")
    
    metrics = run_eval(
        evaluator=evaluator,
        dataset=dataset,
        output_path=args.output,
        workers=args.workers,
        resume=args.resume,
    )
    print_metrics(metrics)


if __name__ == "__main__":
    main()
