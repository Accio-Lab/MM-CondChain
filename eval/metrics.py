"""
Metrics computation for MM-CondChain evaluation.
"""

from typing import List, Dict, Any
from collections import defaultdict


def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Compute evaluation metrics from results.
    
    Args:
        results: List of result dicts, each containing:
            - domain: str
            - true_path_pred: str
            - true_path_gt: str
            - false_path_pred: str
            - false_path_gt: str
            
    Returns:
        Dict of metrics per domain:
        {
            "natural": {"true_acc": 0.x, "false_acc": 0.x, "path_f1": 0.x, "count": n},
            "chart": {...},
            "gui": {...},
            "overall": {...}
        }
    """
    domain_stats = defaultdict(lambda: {
        "true_correct": 0,
        "true_total": 0,
        "false_correct": 0,
        "false_total": 0,
    })
    
    for r in results:
        domain = r.get("domain", "unknown")
        
        if r.get("true_path_pred") is not None and r.get("true_path_gt") is not None:
            domain_stats[domain]["true_total"] += 1
            if r["true_path_pred"] == r["true_path_gt"]:
                domain_stats[domain]["true_correct"] += 1
        
        if r.get("false_path_pred") is not None and r.get("false_path_gt") is not None:
            domain_stats[domain]["false_total"] += 1
            if r["false_path_pred"] == r["false_path_gt"]:
                domain_stats[domain]["false_correct"] += 1
    
    metrics = {}
    
    overall_true_correct = 0
    overall_true_total = 0
    overall_false_correct = 0
    overall_false_total = 0
    
    for domain, stats in domain_stats.items():
        true_acc = stats["true_correct"] / stats["true_total"] if stats["true_total"] > 0 else 0.0
        false_acc = stats["false_correct"] / stats["false_total"] if stats["false_total"] > 0 else 0.0
        
        if true_acc + false_acc > 0:
            path_f1 = 2 * true_acc * false_acc / (true_acc + false_acc)
        else:
            path_f1 = 0.0
        
        metrics[domain] = {
            "true_acc": round(true_acc * 100, 2),
            "false_acc": round(false_acc * 100, 2),
            "path_f1": round(path_f1 * 100, 2),
            "count": stats["true_total"],
        }
        
        overall_true_correct += stats["true_correct"]
        overall_true_total += stats["true_total"]
        overall_false_correct += stats["false_correct"]
        overall_false_total += stats["false_total"]
    
    overall_true_acc = overall_true_correct / overall_true_total if overall_true_total > 0 else 0.0
    overall_false_acc = overall_false_correct / overall_false_total if overall_false_total > 0 else 0.0
    
    if overall_true_acc + overall_false_acc > 0:
        overall_path_f1 = 2 * overall_true_acc * overall_false_acc / (overall_true_acc + overall_false_acc)
    else:
        overall_path_f1 = 0.0
    
    metrics["overall"] = {
        "true_acc": round(overall_true_acc * 100, 2),
        "false_acc": round(overall_false_acc * 100, 2),
        "path_f1": round(overall_path_f1 * 100, 2),
        "count": overall_true_total,
    }
    
    return metrics


def print_metrics(metrics: Dict[str, Dict[str, float]]):
    """
    Pretty print metrics table.
    
    Args:
        metrics: Output from compute_metrics()
    """
    print("\n" + "=" * 70)
    print("MM-CondChain Evaluation Results")
    print("=" * 70)
    print(f"{'Domain':<12} {'True Acc':<12} {'False Acc':<12} {'Path F1':<12} {'Count':<8}")
    print("-" * 70)
    
    domain_order = ["natural", "chart", "gui"]
    for domain in domain_order:
        if domain in metrics:
            m = metrics[domain]
            print(f"{domain:<12} {m['true_acc']:>10.2f}% {m['false_acc']:>10.2f}% {m['path_f1']:>10.2f}% {m['count']:>6}")
    
    print("-" * 70)
    if "overall" in metrics:
        m = metrics["overall"]
        print(f"{'Overall':<12} {m['true_acc']:>10.2f}% {m['false_acc']:>10.2f}% {m['path_f1']:>10.2f}% {m['count']:>6}")
    print("=" * 70 + "\n")
