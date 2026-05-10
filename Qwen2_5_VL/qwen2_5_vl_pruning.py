"""
Qwen2.5-VL Token Pruning Implementation
Adapted from LLaVA StaPrune strategy
"""
import math
import torch
import torch.nn.functional as F


def prune_vision_tokens(hidden_states, attn_scores, tok_keep_ratio, prune_method="prunemerge", selection_strategy="auto"):
    """
    Prune vision tokens based on attention scores.

    Args:
        hidden_states: [N, D] vision features after merger
        attn_scores: attention scores from last full attention layer
        tok_keep_ratio: ratio of tokens to keep (0.0-1.0)
        prune_method: only "prunemerge" supported
        selection_strategy: "auto", "v139g", or "v105"

    Returns:
        pruned_features: [K, D] where K = N * tok_keep_ratio
    """
    if tok_keep_ratio is None or tok_keep_ratio >= 1.0:
        return hidden_states

    if prune_method != "prunemerge":
        raise ValueError(f"Unknown prune_method: {prune_method}")

    N, D = hidden_states.shape
    device = hidden_states.device
    dtype = hidden_states.dtype
    k_total = math.ceil(N * tok_keep_ratio)

    # Normalize attention scores
    attn_scores = attn_scores / (attn_scores.sum() + 1e-8)

    # Auto routing based on retention rate
    if selection_strategy in ["auto", "v112"]:
        if tok_keep_ratio >= 0.25:
            return _v139g_selection(hidden_states, attn_scores, k_total, N, D, device, dtype)
        else:
            return _v105_selection(hidden_states, attn_scores, k_total, N, D, device, dtype)
    elif selection_strategy == "v139g":
        return _v139g_selection(hidden_states, attn_scores, k_total, N, D, device, dtype)
    elif selection_strategy == "v105":
        return _v105_selection(hidden_states, attn_scores, k_total, N, D, device, dtype)
    else:
        raise ValueError(f"Unknown selection_strategy: {selection_strategy}")


def _v105_selection(hidden_states, attn_scores, k_total, N, D, device, dtype):
    """V105: two-stage selection with optional merge for extreme pruning."""
    tok_keep_ratio = k_total / N
    if tok_keep_ratio >= 0.25:
        diversity_ratio = 0.40
    elif tok_keep_ratio >= 0.15:
        diversity_ratio = 0.30
    else:
        diversity_ratio = 0.25

    k_direct = int(k_total * (1 - diversity_ratio))
    k_diverse = k_total - k_direct

    # Stage 1: Direct selection by importance
    direct_indices = torch.topk(attn_scores, k=k_direct).indices.tolist()

    # Stage 2: Greedy max-min diversity
    selected = direct_indices.copy()
    remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
    remaining_mask[direct_indices] = False
    remaining_idx = torch.where(remaining_mask)[0]

    patch_features_norm = F.normalize(hidden_states, dim=-1)

    for _ in range(k_diverse):
        if len(remaining_idx) == 0:
            break
        remaining_feat = patch_features_norm[remaining_idx]
        selected_feat = patch_features_norm[selected]
        similarity = torch.mm(remaining_feat, selected_feat.T)
        distances = 1.0 - similarity
        min_distances = distances.min(dim=-1).values
        best_local_idx = min_distances.argmax()
        best_global_idx = remaining_idx[best_local_idx].item()
        selected.append(best_global_idx)
        remaining_mask[best_global_idx] = False
        remaining_idx = torch.where(remaining_mask)[0]

    final_indices = torch.tensor(selected, device=device)

    # Optional merge for extreme pruning
    if tok_keep_ratio < 0.15:
        k_merge = min(k_total, N - k_total)
        if k_merge > 0:
            _, top_indices = torch.topk(attn_scores, k=min(k_total * 2, N))
            selected_set = set(final_indices.tolist())
            unselected = [idx.item() for idx in top_indices if idx.item() not in selected_set][:k_merge]

            if unselected:
                unselected_feat = hidden_states[unselected]
                unselected_importance = attn_scores[unselected]
                weights = unselected_importance / (unselected_importance.sum() + 1e-8)
                merged_token = (unselected_feat * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)

                selected_features = hidden_states[final_indices]
                return torch.cat([selected_features, merged_token], dim=0)

    final_indices_sorted = torch.sort(final_indices).values
    return hidden_states[final_indices_sorted]


def _v139g_selection(hidden_states, attn_scores, k_total, N, D, device, dtype):
    """V139-G: three-stage selection with spatial coverage."""
    k_stage1 = int(k_total * 0.27)
    k_stage2 = int(k_total * 0.46)
    k_stage3 = k_total - k_stage1 - k_stage2

    patch_features_norm = F.normalize(hidden_states, dim=-1)

    # Stage 1: Top importance
    stage1_indices = torch.topk(attn_scores, k=k_stage1).indices

    # Stage 2: Expand from stage1 pivots
    selected_so_far = stage1_indices.clone()
    all_indices = torch.arange(N, device=device)
    remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
    remaining_mask[selected_so_far] = False
    remaining_indices = all_indices[remaining_mask]

    stage2_selected = []
    for pivot_idx in stage1_indices:
        if len(remaining_indices) == 0 or len(stage2_selected) >= k_stage2:
            break
        pivot_feat = patch_features_norm[pivot_idx]
        remaining_feat = patch_features_norm[remaining_indices]
        similarity = torch.mv(remaining_feat, pivot_feat)
        k_expand = min(2 if len(stage2_selected) < 52 else 1, len(remaining_indices), k_stage2 - len(stage2_selected))
        if k_expand > 0:
            top_similar = torch.topk(similarity, k=k_expand).indices
            selected_tokens = remaining_indices[top_similar]
            stage2_selected.extend(selected_tokens.tolist())
            remaining_mask[selected_tokens] = False
            remaining_indices = all_indices[remaining_mask]

    stage2_indices = torch.tensor(stage2_selected[:k_stage2], device=device)
    selected_so_far = torch.cat([selected_so_far, stage2_indices])

    # Stage 3: Max-min diversity
    remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
    remaining_mask[selected_so_far] = False
    remaining_indices = all_indices[remaining_mask]

    selected_features_norm = patch_features_norm[selected_so_far]
    remaining_features_norm = patch_features_norm[remaining_indices]

    stage3_selected = []
    for _ in range(k_stage3):
        if len(remaining_features_norm) == 0:
            break
        similarity = torch.mm(remaining_features_norm, selected_features_norm.t())
        max_similarity = similarity.max(dim=-1).values
        best_idx_local = max_similarity.argmin()
        best_idx_global = remaining_indices[best_idx_local]
        stage3_selected.append(best_idx_global.item())
        selected_features_norm = torch.cat([selected_features_norm, remaining_features_norm[best_idx_local:best_idx_local+1]], dim=0)
        remaining_mask_local = torch.ones(len(remaining_indices), dtype=torch.bool, device=device)
        remaining_mask_local[best_idx_local] = False
        remaining_indices = remaining_indices[remaining_mask_local]
        remaining_features_norm = remaining_features_norm[remaining_mask_local]

    stage3_indices = torch.tensor(stage3_selected, device=device)
    final_indices = torch.cat([stage1_indices, stage2_indices, stage3_indices])
    final_indices_sorted = torch.sort(final_indices).values

    return hidden_states[final_indices_sorted]
