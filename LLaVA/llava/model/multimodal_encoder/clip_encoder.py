import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class CLIPVisionTower(nn.Module):
    def __init__(
        self,
        vision_tower,
        args,
        tok_keep_ratio=None,
        prune_method="prunemerge",
        selection_strategy="auto",
        ablation_mode="full",
        delay_load=False,
    ):
        """
        FasterVLM token pruning with V139-G and V105 only.

        selection_strategy:
          - "auto" or "v112": retention >= 25% -> V139-G, retention < 25% -> V105
          - "v139g": force V139-G
          - "v105": force V105
        """
        super().__init__()

        self.is_loaded = False
        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        self.tok_keep_ratio = tok_keep_ratio
        self.prune_method = prune_method
        self.selection_strategy = selection_strategy or "auto"
        self.ablation_mode = "full" if ablation_mode is None else str(ablation_mode).strip()
        self.enabled_modules = self._resolve_enabled_modules(self.ablation_mode)

        # Use last N layers to fuse CLS attention.
        self.num_layers_for_attn = 4

        if not delay_load:
            self.load_model()
        elif getattr(args, "unfreeze_mm_vision_tower", False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        # Use eager attention to support output_attentions=True.
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
            attn_implementation="eager",
        )
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def _fuse_cls_attention(self, all_attentions):
        num_layers = len(all_attentions)
        start_layer = max(0, num_layers - self.num_layers_for_attn)
        fused_attn = None
        for layer_idx in range(start_layer, num_layers):
            attn = all_attentions[layer_idx]
            cls_attn = torch.mean(attn[:, :, 0, 1:], dim=1)
            weight = (layer_idx - start_layer + 1) / self.num_layers_for_attn
            fused_attn = cls_attn * weight if fused_attn is None else fused_attn + cls_attn * weight
        fused_attn = fused_attn / (fused_attn.sum(dim=-1, keepdim=True) + 1e-8)
        return fused_attn

    def _resolve_enabled_modules(self, ablation_mode):
        full_modules = {"A", "B", "C", "D", "R"}
        mode = "full" if ablation_mode is None else str(ablation_mode).strip().lower()

        alias_to_enabled = {
            "": full_modules,
            "none": full_modules,
            "baseline": full_modules,
            "full": full_modules,
            "full_method": full_modules,
            "no_aux": {"B", "C", "D", "R"},
            "no_aip": {"B", "C", "D", "R"},
            "wo_importance_prioritization": {"B", "C", "D", "R"},
            "wo_auxiliary_variable_guided_importance_prioritization": {"B", "C", "D", "R"},
            "wo_aip": {"B", "C", "D", "R"},
            "no_neighbor": {"A", "C", "D", "R"},
            "wo_conditional_neighborhood_expansion": {"A", "C", "D", "R"},
            "no_dispersion": {"A", "B", "D", "R"},
            "wo_greedy_dispersion": {"A", "B", "D", "R"},
            "no_secb": {"A", "D", "R"},
            "wo_secb": {"A", "D", "R"},
            "no_spatial": {"A", "B", "C", "R"},
            "wo_spatial_coverage_refinement": {"A", "B", "C", "R"},
            "no_scr": {"A", "B", "C", "R"},
            "wo_scr": {"A", "B", "C", "R"},
            "no_residual": {"A", "B", "C", "D"},
            "wo_residual_aggregation": {"A", "B", "C", "D"},
            "no_cra": {"A", "B", "C", "D"},
            "wo_cra": {"A", "B", "C", "D"},
        }
        if mode in alias_to_enabled:
            return set(alias_to_enabled[mode])

        module_aliases = {
            "a": {"A"},
            "aip": {"A"},
            "importance": {"A"},
            "aux": {"A"},
            "importance_prioritization": {"A"},
            "auxiliary_variable_guided_importance_prioritization": {"A"},
            "b": {"B"},
            "e": {"B"},
            "expansion": {"B"},
            "neighbor": {"B"},
            "neighborhood": {"B"},
            "conditional_neighborhood_expansion": {"B"},
            "c": {"C"},
            "dispersion": {"C"},
            "diversity": {"C"},
            "greedy_dispersion": {"C"},
            "global_coverage": {"C"},
            "secb": {"B", "C"},
            "d": {"D"},
            "s": {"D"},
            "scr": {"D"},
            "spatial": {"D"},
            "spatial_refinement": {"D"},
            "spatial_coverage_refinement": {"D"},
            "r": {"R"},
            "cra": {"R"},
            "residual": {"R"},
            "residual_aggregation": {"R"},
        }
        tokens = [tok for tok in re.split(r"[+,|/;:\s]+", mode) if tok]
        if not tokens:
            return set(full_modules)

        enabled_modules = set()
        for token in tokens:
            if token == "all":
                enabled_modules.update(full_modules)
                continue
            mapped = module_aliases.get(token)
            if mapped is None:
                raise ValueError(
                    f"Unknown ablation_mode/module combo: {ablation_mode}. "
                    "Use AIP/SECB/SCR/CRA or the legacy A/B/C/D/R aliases."
                )
            enabled_modules.update(mapped)
        return enabled_modules

    def _has_module(self, module_name):
        return module_name in self.enabled_modules

    def _uniform_flat_indices(self, total_tokens, k, device):
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if k >= total_tokens:
            return torch.arange(total_tokens, dtype=torch.long, device=device)
        positions = (((torch.arange(k, device=device, dtype=torch.float32) + 0.5) * total_tokens) / k).floor()
        return positions.long().clamp(max=total_tokens - 1)

    def _select_batch_indices(self, scores, k, device):
        if k <= 0:
            return torch.empty(scores.shape[0], 0, dtype=torch.long, device=device)
        if not self._has_module("A"):
            shared_indices = self._uniform_flat_indices(scores.shape[-1], k, device)
            return shared_indices.unsqueeze(0).expand(scores.shape[0], -1)
        return torch.topk(scores, k=k, dim=-1).indices

    def _select_stage1_indices(self, attention, k, total_tokens, device):
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if not self._has_module("A"):
            return self._uniform_flat_indices(total_tokens, k, device)
        return torch.topk(attention, k=k).indices

    def _fill_remaining_with_importance(self, attention, selected_indices, total_tokens, target_count, device):
        current_count = selected_indices.numel()
        if current_count >= target_count:
            return selected_indices[:target_count]

        remaining_mask = torch.ones(total_tokens, dtype=torch.bool, device=device)
        if current_count > 0:
            remaining_mask[selected_indices] = False
        remaining_indices = torch.arange(total_tokens, device=device)[remaining_mask]
        if remaining_indices.numel() == 0:
            return selected_indices

        need = min(target_count - current_count, remaining_indices.numel())
        if not self._has_module("A"):
            extra_indices = remaining_indices[self._uniform_flat_indices(remaining_indices.numel(), need, device)]
        else:
            extra_scores = attention[remaining_indices]
            extra_indices = remaining_indices[torch.topk(extra_scores, k=need).indices]

        if current_count == 0:
            return extra_indices
        return torch.cat([selected_indices, extra_indices], dim=0)

    def _select_region_candidate(self, region_candidates_tensor, attention):
        if region_candidates_tensor.numel() == 0:
            return None
        if not self._has_module("A"):
            return region_candidates_tensor[region_candidates_tensor.numel() // 2]
        return region_candidates_tensor[torch.argmax(attention[region_candidates_tensor])]

    def _greedy_max_min_diversity_v105(self, patch_features, importance_scores, direct_indices, k_diverse, device):
        """Greedy max-min diversity selection (DivPrune style)."""
        B, N, _ = patch_features.shape
        selected_indices_list = []

        for b in range(B):
            selected = direct_indices[b].tolist()
            remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
            remaining_mask[direct_indices[b]] = False
            remaining_idx = torch.where(remaining_mask)[0]

            for _ in range(k_diverse):
                if len(remaining_idx) == 0:
                    break
                remaining_feat_norm = F.normalize(patch_features[b, remaining_idx], dim=-1)
                selected_feat_norm = F.normalize(patch_features[b, selected], dim=-1)
                similarity = torch.mm(remaining_feat_norm, selected_feat_norm.T)
                distances = 1.0 - similarity
                min_distances = distances.min(dim=-1).values
                best_local_idx = min_distances.argmax()
                best_global_idx = remaining_idx[best_local_idx].item()
                selected.append(best_global_idx)
                remaining_mask[best_global_idx] = False
                remaining_idx = torch.where(remaining_mask)[0]

            selected_indices_list.append(torch.tensor(selected, device=device))

        max_selected = max(len(s) for s in selected_indices_list)
        final_indices = torch.zeros(B, max_selected, dtype=torch.long, device=device)
        for b, s in enumerate(selected_indices_list):
            final_indices[b, : len(s)] = s
        return final_indices

    def _select_and_merge_tokens_v105(self, patch_features, importance_scores, selected_indices, k_total, D, device, dtype):
        """Merge high-importance unselected tokens for extreme pruning (<15%)."""
        B, N, _ = patch_features.shape
        selected_features = torch.gather(
            patch_features, 1, selected_indices.unsqueeze(-1).expand(-1, -1, D)
        )
        k_merge_candidates = min(k_total, N - k_total)
        if k_merge_candidates == 0:
            return selected_features

        _, top_indices = torch.topk(importance_scores, k=min(k_total * 2, N), dim=-1)
        merged_tokens_list = []
        for b in range(B):
            selected_set = set(selected_indices[b].tolist())
            unselected = [idx.item() for idx in top_indices[b] if idx.item() not in selected_set]
            unselected = unselected[:k_merge_candidates]
            if unselected:
                unselected_feat = patch_features[b, unselected]
                unselected_importance = importance_scores[b, unselected]
                weights = unselected_importance / (unselected_importance.sum() + 1e-8)
                merged_token = (unselected_feat * weights.unsqueeze(-1)).sum(dim=0, keepdim=True)
                merged_tokens_list.append(merged_token)
            else:
                merged_tokens_list.append(torch.zeros(1, D, device=device, dtype=dtype))

        merged_tokens = torch.cat(merged_tokens_list, dim=0).unsqueeze(1)
        return torch.cat([selected_features, merged_tokens], dim=1)

    def _max_min_diversity_selection(self, patch_features, selected_indices, remaining_indices, k, device):
        """Max-min diversity selection used by V139-G stage 3."""
        selected_features = patch_features[selected_indices]
        remaining_features = patch_features[remaining_indices]
        selected_features_norm = F.normalize(selected_features, dim=-1)
        remaining_features_norm = F.normalize(remaining_features, dim=-1)

        diversity_selected = []
        for _ in range(k):
            if len(remaining_features_norm) == 0:
                break
            similarity = torch.mm(remaining_features_norm, selected_features_norm.t())
            max_similarity = similarity.max(dim=-1).values
            best_idx_local = max_similarity.argmin()
            best_idx_global = remaining_indices[best_idx_local]
            diversity_selected.append(best_idx_global.item())
            selected_features_norm = torch.cat(
                [selected_features_norm, remaining_features_norm[best_idx_local : best_idx_local + 1]],
                dim=0,
            )
            remaining_mask = torch.ones(len(remaining_indices), dtype=torch.bool, device=device)
            remaining_mask[best_idx_local] = False
            remaining_indices = remaining_indices[remaining_mask]
            remaining_features_norm = remaining_features_norm[remaining_mask]

        return torch.tensor(diversity_selected, device=device, dtype=torch.long)

    def _v105_selection(self, patch_features, attn_scores, images, N, grid_size, D, device, dtype):
        """V105: simple two-stage selection with optional token merge."""
        k_total = math.ceil(N * self.tok_keep_ratio)
        hybrid_scores = attn_scores
        hybrid_scores = hybrid_scores / (hybrid_scores.sum(dim=-1, keepdim=True) + 1e-8)

        if not self._has_module("C"):
            diversity_ratio = 0.0
        elif self.tok_keep_ratio >= 0.25:
            diversity_ratio = 0.40
        elif self.tok_keep_ratio >= 0.15:
            diversity_ratio = 0.30
        else:
            diversity_ratio = 0.25

        k_direct = int(k_total * (1 - diversity_ratio))
        k_diverse = k_total - k_direct

        direct_indices = self._select_batch_indices(hybrid_scores, k=k_direct, device=device)
        if k_diverse > 0:
            final_indices = self._greedy_max_min_diversity_v105(
                patch_features, hybrid_scores, direct_indices, k_diverse, device
            )
        else:
            final_indices = direct_indices

        if self.tok_keep_ratio < 0.15 and self._has_module("R"):
            selected_features = self._select_and_merge_tokens_v105(
                patch_features, hybrid_scores, final_indices, k_total, D, device, dtype
            )
        else:
            topk_sorted = torch.sort(final_indices, dim=-1).values
            selected_features = torch.gather(
                patch_features, 1, topk_sorted.unsqueeze(-1).expand(-1, -1, D)
            )
        return selected_features

    def _v139_post_selection_refinement(self, patch_features, attn_scores, images, N, grid_size, D, device, dtype):
        """V139-G: V111 selection plus spatial coverage refinement."""
        B = patch_features.shape[0]
        k_total = math.ceil(N * self.tok_keep_ratio)
        use_importance = self._has_module("A")
        use_neighbor = self._has_module("B")
        use_dispersion = self._has_module("C")
        use_spatial = self._has_module("D")

        base_stage1 = max(1, int(k_total * 0.27))
        base_stage2 = int(k_total * 0.46)

        if use_neighbor and use_dispersion:
            k_stage1 = min(base_stage1, max(1, k_total - 1))
            k_stage2 = min(base_stage2, k_total - k_stage1)
            k_stage3 = k_total - k_stage1 - k_stage2
        elif use_neighbor:
            k_stage1 = min(base_stage1, max(1, k_total - 1))
            k_stage2 = k_total - k_stage1
            k_stage3 = 0
        elif use_dispersion:
            k_stage1 = min(base_stage1 if use_importance else 1, k_total)
            k_stage2 = 0
            k_stage3 = k_total - k_stage1
        else:
            k_stage1 = k_total
            k_stage2 = 0
            k_stage3 = 0

        patch_features_norm = patch_features / (patch_features.norm(dim=-1, keepdim=True) + 1e-8)
        selected_features_list = []

        for b in range(B):
            attention = attn_scores[b]
            stage1_indices = self._select_stage1_indices(attention, k_stage1, N, device)

            selected_so_far = stage1_indices.clone()
            all_indices = torch.arange(N, device=device)
            remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
            remaining_mask[selected_so_far] = False
            remaining_indices = all_indices[remaining_mask]

            stage2_selected = []
            if k_stage2 > 0:
                for pivot_pos, pivot_idx in enumerate(stage1_indices):
                    if len(remaining_indices) == 0 or len(stage2_selected) >= k_stage2:
                        break
                    pivot_feat = patch_features_norm[b][pivot_idx]
                    remaining_feat = patch_features_norm[b][remaining_indices]
                    similarity = torch.mv(remaining_feat, pivot_feat)
                    remaining_quota = k_stage2 - len(stage2_selected)
                    if use_dispersion:
                        proposed_expand = 2 if len(stage2_selected) < 52 else 1
                    else:
                        remaining_pivots = max(1, len(stage1_indices) - pivot_pos)
                        proposed_expand = max(1, math.ceil(remaining_quota / remaining_pivots))
                    k_expand = min(proposed_expand, len(remaining_indices), remaining_quota)
                    if k_expand > 0:
                        top_similar = torch.topk(similarity, k=k_expand).indices
                        selected_tokens = remaining_indices[top_similar]
                        stage2_selected.extend(selected_tokens.tolist())
                        remaining_mask[selected_tokens] = False
                        remaining_indices = all_indices[remaining_mask]

            stage2_indices = torch.tensor(stage2_selected[:k_stage2], device=device, dtype=torch.long)
            selected_so_far = torch.cat([selected_so_far, stage2_indices])

            remaining_mask = torch.ones(N, dtype=torch.bool, device=device)
            remaining_mask[selected_so_far] = False
            remaining_indices = all_indices[remaining_mask]
            if use_dispersion and k_stage3 > 0:
                k_stage3 = k_total - selected_so_far.numel()
            if use_dispersion and k_stage3 > 0:
                stage3_indices = self._max_min_diversity_selection(
                    patch_features_norm[b], selected_so_far, remaining_indices, k_stage3, device
                )
            else:
                stage3_indices = torch.empty(0, dtype=torch.long, device=device)

            final_indices = torch.cat([stage1_indices, stage2_indices, stage3_indices])
            if final_indices.numel() < k_total:
                final_indices = self._fill_remaining_with_importance(
                    attention, final_indices, N, k_total, device
                )

            # Spatial coverage refinement (V139-G)
            if use_spatial:
                num_regions = 4
                region_size = max(1, grid_size // num_regions)
                coverage_map = torch.zeros(num_regions, num_regions, device=device)
                for idx in final_indices:
                    h = idx // grid_size
                    w = idx % grid_size
                    region_h = h // region_size
                    region_w = w // region_size
                    if region_h < num_regions and region_w < num_regions:
                        coverage_map[region_h, region_w] += 1

                weak_threshold = 5.0
                weak_regions = []
                for i in range(num_regions):
                    for j in range(num_regions):
                        if coverage_map[i, j] < weak_threshold:
                            weak_regions.append((i, j))

                if weak_regions:
                    max_swaps = min(5, len(weak_regions))
                    stage3_marginality = []
                    selected_features = patch_features_norm[b][final_indices]
                    replacement_pool = stage3_indices if stage3_indices.numel() > 0 else final_indices
                    for replacement_idx in replacement_pool:
                        replacement_feat = patch_features_norm[b][replacement_idx]
                        similarities = torch.mv(selected_features, replacement_feat)
                        min_sim = similarities.min().item()
                        stage3_marginality.append((replacement_idx.item(), min_sim))
                    stage3_marginality.sort(key=lambda x: x[1])
                    most_marginal = [idx for idx, _ in stage3_marginality[:max_swaps]]

                    replacements = []
                    final_indices_list = final_indices.tolist()
                    final_indices_set = set(final_indices_list)
                    for region_h, region_w in weak_regions[:max_swaps]:
                        h_start = region_h * region_size
                        h_end = (region_h + 1) * region_size
                        w_start = region_w * region_size
                        w_end = (region_w + 1) * region_size
                        region_candidates = []
                        for h in range(h_start, h_end):
                            for w in range(w_start, w_end):
                                idx = h * grid_size + w
                                if idx not in final_indices_set and idx < N:
                                    region_candidates.append(idx)
                        if region_candidates:
                            region_candidates_tensor = torch.tensor(region_candidates, device=device)
                            best_candidate = self._select_region_candidate(region_candidates_tensor, attention)
                            if best_candidate is not None:
                                replacements.append(best_candidate.item())

                    for marginal_idx, replacement_idx in zip(most_marginal[: len(replacements)], replacements):
                        if marginal_idx in final_indices_list:
                            idx_pos = final_indices_list.index(marginal_idx)
                            final_indices_list[idx_pos] = replacement_idx

                    final_indices = torch.tensor(final_indices_list, device=device)

            final_indices_sorted = torch.sort(final_indices).values
            selected_features = torch.gather(
                patch_features[b], 0, final_indices_sorted.unsqueeze(-1).expand(-1, D)
            )
            selected_features_list.append(selected_features)

        return torch.stack(selected_features_list, dim=0)

    def _v112_retention_adaptive(self, patch_features, attn_scores, images, N, grid_size, D, device, dtype):
        """Auto router for V139-G / V105 based on retention."""
        k_total = math.ceil(N * self.tok_keep_ratio)
        keep_ratio = self.tok_keep_ratio

        if keep_ratio >= 0.25:
            if not hasattr(self, "_printed_delegate_v139g"):
                print(f"\n[Auto Router] Retention: {keep_ratio:.1%} ({k_total}/{N} tokens)")
                print("  Strategy: Delegate to V139-G")
                self._printed_delegate_v139g = True
            return self._v139_post_selection_refinement(
                patch_features, attn_scores, images, N, grid_size, D, device, dtype
            )

        if not hasattr(self, "_printed_delegate_v105"):
            print(f"\n[Auto Router] Retention: {keep_ratio:.1%} ({k_total}/{N} tokens)")
            print("  Strategy: Delegate to V105")
            self._printed_delegate_v105 = True
        return self._v105_selection(patch_features, attn_scores, images, N, grid_size, D, device, dtype)

    def feature_select_prunemerge(self, image_forward_outs, images=None):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        all_attentions = image_forward_outs.attentions

        B, num_tokens_with_cls, D = image_features.shape
        N = num_tokens_with_cls - 1
        device = image_features.device
        dtype = image_features.dtype
        grid_size = int(math.sqrt(N))

        patch_features = image_features[:, 1:, :]
        attn_scores = self._fuse_cls_attention(all_attentions)

        if self.selection_strategy in ["auto", "v112"]:
            selected_features = self._v112_retention_adaptive(
                patch_features, attn_scores, images, N, grid_size, D, device, dtype
            )
        elif self.selection_strategy == "v139g":
            selected_features = self._v139_post_selection_refinement(
                patch_features, attn_scores, images, N, grid_size, D, device, dtype
            )
        elif self.selection_strategy == "v105":
            selected_features = self._v105_selection(
                patch_features, attn_scores, images, N, grid_size, D, device, dtype
            )
        else:
            raise ValueError(f"Unknown selection_strategy: {self.selection_strategy}")

        if self.select_feature == "patch":
            return selected_features
        if self.select_feature == "cls_patch":
            return torch.cat([image_features[:, 0:1, :], selected_features], dim=1)
        return selected_features

    def feature_select(self, image_forward_outs, images=None, texts=None):
        if self.tok_keep_ratio is None or self.tok_keep_ratio >= 1.0:
            feat = image_forward_outs.hidden_states[self.select_layer]
            return feat[:, 1:] if self.select_feature == "patch" else feat
        if self.prune_method != "prunemerge":
            raise ValueError(f"Unknown prune_method: {self.prune_method}")
        return self.feature_select_prunemerge(image_forward_outs, images)

    @torch.no_grad()
    def forward(self, images, texts=None):
        if type(images) is list:
            return [
                self.feature_select(
                    self.vision_tower(
                        img.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_attentions=True,
                        output_hidden_states=True,
                    ),
                    img.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    texts,
                ).to(img.dtype)
                for img in images
            ]

        orig_dtype = images.dtype
        images = images.to(device=self.device, dtype=self.dtype)
        out = self.vision_tower(images, output_attentions=True, output_hidden_states=True)
        return self.feature_select(out, images, texts).to(orig_dtype)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config if self.is_loaded else self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size


class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(
        self,
        vision_tower,
        args,
        tok_keep_ratio=None,
        prune_method="prunemerge",
        selection_strategy="auto",
        ablation_mode="full",
        delay_load=False,
    ):
        super().__init__(
            vision_tower,
            args,
            tok_keep_ratio=tok_keep_ratio,
            prune_method=prune_method,
            selection_strategy=selection_strategy,
            ablation_mode=ablation_mode,
            delay_load=delay_load,
        )
        self.s2_scales = list(map(int, getattr(args, "s2_scales", "336,672,1008").split(",")))
        self.s2_scales.sort()
        self.s2_split_size, self.s2_image_size = self.s2_scales[0], self.s2_scales[-1]
        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError as exc:
            raise ImportError("s2wrapper not found") from exc
        self.multiscale_forward = multiscale_forward
        if not delay_load or getattr(args, "unfreeze_mm_vision_tower", False):
            self.image_processor.size["shortest_edge"] = self.s2_image_size
            self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            return
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.vision_tower_name,
            device_map=device_map,
            attn_implementation="eager",
        )
        self.vision_tower.requires_grad_(False)
        self.image_processor.size["shortest_edge"] = self.s2_image_size
        self.image_processor.crop_size["height"] = self.image_processor.crop_size["width"] = self.s2_image_size
        self.is_loaded = True

    @torch.no_grad()
    def forward_feature(self, images, texts=None):
        orig_dtype = images.dtype
        images = images.to(device=self.device, dtype=self.dtype)
        out = self.vision_tower(images, output_attentions=True, output_hidden_states=True)
        return self.feature_select(out, images, texts).to(orig_dtype)

    @torch.no_grad()
    def forward(self, images, texts=None):
        if type(images) is list:
            return [
                self.multiscale_forward(
                    lambda x: self.forward_feature(x, texts=None),
                    img.unsqueeze(0),
                    img_sizes=self.s2_scales,
                    max_split_size=self.s2_split_size,
                )
                for img in images
            ]
        return self.multiscale_forward(
            lambda x: self.forward_feature(x, texts=texts),
            images,
            img_sizes=self.s2_scales,
            max_split_size=self.s2_split_size,
        )

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
