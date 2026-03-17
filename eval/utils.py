"""
Utility functions for MM-CondChain evaluation.
"""

import re
import base64
import mimetypes
from typing import Optional


BOXED_RE = re.compile(r"\\boxed\s*\{\s*([^}]*)\s*\}")
LABEL_RE = re.compile(r"\b([A-Z]\d+|None)\b")


def build_eval_prompt(instruction: str) -> str:
    """
    Build evaluation prompt with boxed answer format.
    
    Args:
        instruction: The full instruction from true_path or false_path
        
    Returns:
        Formatted prompt string
    """
    return f"""\
Please answer the following question based on the image(s):
{instruction}

Please select the correct option from all choices.
Output your answer in the format: \\boxed{{letter-number}}
Replace letter-number with the option you choose. If you believe there is no correct answer, output \\boxed{{None}}.
"""


def parse_answer(text: str) -> str:
    """
    Parse model output to extract the answer.
    
    Priority:
    1. Extract from \\boxed{...} (last match)
    2. Fallback to pattern matching (A1, B3, G2, None, etc.)
    
    Args:
        text: Raw model output text
        
    Returns:
        Parsed answer string (e.g., "A1", "G2", "None")
    """
    t = text or ""
    
    boxed = BOXED_RE.findall(t)
    if boxed:
        ans = boxed[-1].strip().replace(" ", "")
        return "None" if ans.lower() in ("none", "null", "nan", "") else ans
    
    m = LABEL_RE.findall(t)
    if m:
        ans = m[-1].strip().replace(" ", "")
        return "None" if ans.lower() == "none" else ans
    
    return "None"


def encode_image_to_base64(image_path: str) -> str:
    """
    Encode an image file to base64 string.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def guess_mime(image_path: str) -> str:
    """
    Guess MIME type from file extension.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        MIME type string (defaults to "image/jpeg")
    """
    mt, _ = mimetypes.guess_type(image_path)
    return mt or "image/jpeg"


def build_message_content(
    instruction: str,
    image_path: Optional[str] = None,
    image_paths: Optional[list] = None,
) -> Optional[list]:
    """
    Build OpenAI-compatible message content with images.
    
    Args:
        instruction: The instruction text
        image_path: Single image path (for Natural/Chart)
        image_paths: List of image paths (for GUI multi-frame)
        
    Returns:
        List of content items for OpenAI API, or None if no images
    """
    prompt = build_eval_prompt(instruction)
    content = []
    
    if image_paths and len(image_paths) > 0:
        for i, img_path in enumerate(image_paths):
            img_b64 = encode_image_to_base64(img_path)
            mime = guess_mime(img_path)
            content.append({"type": "text", "text": f"Image {i + 1}:"})
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{img_b64}"}
            })
        content.append({"type": "text", "text": f"\n\n{prompt}"})
    elif image_path:
        img_b64 = encode_image_to_base64(image_path)
        mime = guess_mime(image_path)
        content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": prompt},
        ]
    else:
        return None
    
    return content
