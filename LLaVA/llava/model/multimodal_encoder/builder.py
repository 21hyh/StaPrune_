import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2


def build_vision_tower(vision_tower_cfg, tok_keep_ratio=None, prune_method="prunemerge", selection_strategy="auto", ablation_mode="full", **kwargs):
    """
    构建视觉编码器

    Args:
        vision_tower_cfg: 视觉塔配置
        tok_keep_ratio: token保留比例 (0.0-1.0)
        prune_method: 剪枝方法，仅保留 "prunemerge"
        selection_strategy: 策略选择（默认auto）
            - "auto"/"v112": retention >= 25% 使用V139-G，retention < 25% 使用V105
            - "v139g": 固定使用V139-G
            - "v105": 固定使用V105
        ablation_mode: 逐模块消融模式
        **kwargs: 其他参数
    """
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(
                vision_tower,
                args=vision_tower_cfg,
                tok_keep_ratio=tok_keep_ratio,
                prune_method=prune_method,
                selection_strategy=selection_strategy,
                ablation_mode=ablation_mode,
                **kwargs,
            )
        else:
            return CLIPVisionTower(
                vision_tower,
                args=vision_tower_cfg,
                tok_keep_ratio=tok_keep_ratio,
                prune_method=prune_method,
                selection_strategy=selection_strategy,
                ablation_mode=ablation_mode,
                **kwargs,
            )

    raise ValueError(f'Unknown vision tower: {vision_tower}')
