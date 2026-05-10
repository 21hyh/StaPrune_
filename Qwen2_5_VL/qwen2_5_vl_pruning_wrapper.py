"""
Qwen2.5-VL Vision Encoder with Token Pruning
Wraps the vision encoder to add pruning capability
"""
import torch
from .qwen2_5_vl_pruning import prune_vision_tokens


def apply_pruning_to_vision_encoder(vision_model, tok_keep_ratio=None, prune_method="prunemerge", selection_strategy="auto"):
    """
    Configure vision encoder for token pruning.

    Args:
        vision_model: Qwen2_5_VisionTransformerPretrainedModel instance
        tok_keep_ratio: ratio of tokens to keep (0.0-1.0)
        prune_method: only "prunemerge" supported
        selection_strategy: "auto", "v139g", or "v105"
    """
    if tok_keep_ratio is None or tok_keep_ratio >= 1.0:
        return vision_model

    # Set pruning config - the vision encoder has built-in pruning logic
    vision_model.tok_keep_ratio = tok_keep_ratio
    vision_model.prune_method = prune_method
    vision_model.selection_strategy = selection_strategy

    return vision_model
