#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False, **kwargs):
    # 提取Token剪枝参数
    tok_keep_ratio = kwargs.pop("tok_keep_ratio", None)
    prune_method = kwargs.pop("prune_method", "prunemerge")
    selection_strategy = kwargs.pop("selection_strategy", "auto")  # 默认使用自动路由（≥25%走V139-G，<25%走V105）
    ablation_mode = kwargs.pop("ablation_mode", "full")

    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        if 'torch_dtype' not in kwargs and 'dtype' not in kwargs:
            kwargs['torch_dtype'] = torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            # Remove visual_token_num from kwargs as it's not a model parameter
            kwargs.pop('visual_token_num', None)

            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

                # Check if this is llava_next format (by detecting config.json)
                import json
                config_path = os.path.join(model_path, 'config.json')
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)

                is_llava_next = config_dict.get('model_type') == 'llava_next'

                if is_llava_next:
                    # llava_next format requires special handling
                    print(f"Detected llava_next format config, setting up mm_vision_tower...")

                    # Load the model using from_pretrained which will handle device mapping
                    # But state_dict keys won't match, so we'll load with strict=False and fix manually
                    print("Loading model (keys will be remapped)...")

                    # Temporarily rename safetensors files to load them with remapped keys
                    # First, load state_dict from files and remap
                    from safetensors import safe_open
                    import glob
                    import tempfile

                    # Create a temporary directory for remapped weights
                    temp_dir = tempfile.mkdtemp(prefix="llava_next_remap_")
                    print(f"Creating remapped weights in {temp_dir}...")

                    # Load all safetensors files and remap keys
                    safetensors_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))

                    weight_map = {}  # For index file
                    for idx, safetensors_file in enumerate(safetensors_files, 1):
                        print(f"  Remapping {os.path.basename(safetensors_file)}...")
                        state_dict = {}

                        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                tensor = f.get_tensor(key)
                                # Remap keys: remove 'language_model.' prefix
                                if key.startswith('language_model.'):
                                    new_key = key[len('language_model.'):]
                                    state_dict[new_key] = tensor
                                    weight_map[new_key] = f"model-{idx:05d}-of-{len(safetensors_files):05d}.safetensors"
                                else:
                                    state_dict[key] = tensor
                                    weight_map[key] = f"model-{idx:05d}-of-{len(safetensors_files):05d}.safetensors"

                        # Save remapped state_dict to temporary file
                        temp_file = os.path.join(temp_dir, f"model-{idx:05d}-of-{len(safetensors_files):05d}.safetensors")
                        from safetensors.torch import save_file
                        save_file(state_dict, temp_file, metadata={"format": "pt"})

                    # Create index file
                    import json
                    index_data = {
                        "metadata": {"total_size": 0},
                        "weight_map": weight_map
                    }
                    with open(os.path.join(temp_dir, "model.safetensors.index.json"), 'w') as f:
                        json.dump(index_data, f, indent=2)

                    # Copy config and other files
                    import shutil
                    for file in ['tokenizer_config.json', 'tokenizer.model', 'special_tokens_map.json']:
                        src = os.path.join(model_path, file)
                        if os.path.exists(src):
                            shutil.copy(src, os.path.join(temp_dir, file))

                    # Modify config to use llava_llama model type
                    config_path = os.path.join(temp_dir, 'config.json')
                    temp_config = config_dict.get('text_config', {}).copy()
                    temp_config['model_type'] = 'llama'  # Use llama instead of llava_next
                    temp_config['architectures'] = ['LlavaLlamaForCausalLM']
                    # Preserve important fields from original config
                    for key in ['vision_config', 'image_grid_pinpoints', 'vision_feature_layer', 'vision_feature_select_strategy']:
                        if key in config_dict:
                            temp_config[key] = config_dict[key]

                    with open(config_path, 'w') as f:
                        json.dump(temp_config, f, indent=2)

                    # Remove visual_token_num from kwargs as it's not a model parameter
                    kwargs.pop('visual_token_num', None)

                    # Now load model from temporary directory
                    print("Loading model from remapped weights...")
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        temp_dir,
                        low_cpu_mem_usage=True,
                        tok_keep_ratio=tok_keep_ratio,
                        prune_method=prune_method,
                        selection_strategy=selection_strategy,
                        ablation_mode=ablation_mode,
                        **kwargs
                    )
                    print("Model loaded successfully from remapped weights")

                    # Clean up temporary directory
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                        print(f"Cleaned up temporary directory")

                    # Manually add mm_vision_tower related configs that llava_next doesn't have
                    # These are needed for get_vision_tower() and prepare_inputs_labels_for_multimodal()
                    model.config.mm_vision_tower = "openai/clip-vit-large-patch14-336"
                    model.config.mm_vision_select_layer = getattr(model.config, "vision_feature_layer", -2)
                    model.config.mm_vision_select_feature = "patch"
                    model.config.mm_patch_merge_type = "spatial_unpad"
                    model.config.mm_projector_type = "mlp2x_gelu"
                    model.config.image_aspect_ratio = "anyres"

                    # Extract hidden_size from vision_config
                    if hasattr(model.config, 'vision_config'):
                        if isinstance(model.config.vision_config, dict):
                            model.config.mm_hidden_size = model.config.vision_config['hidden_size']
                        else:
                            model.config.mm_hidden_size = model.config.vision_config.hidden_size

                    # Manually build and initialize vision tower since it wasn't created during __init__
                    # (because config didn't have mm_vision_tower at that time)
                    from llava.model.multimodal_encoder.builder import build_vision_tower
                    from llava.model.multimodal_projector.builder import build_vision_projector

                    print("Manually initializing vision tower for llava_next...")
                    vision_tower = build_vision_tower(
                        model.config,
                        tok_keep_ratio=tok_keep_ratio,
                        prune_method=prune_method,
                        selection_strategy=selection_strategy,
                        ablation_mode=ablation_mode,
                        delay_load=True
                    )
                    model.get_model().vision_tower = vision_tower

                    print("Manually initializing mm_projector for llava_next...")
                    language_model_device = next(model.parameters()).device

                    model.get_model().mm_projector = build_vision_projector(model.config)
                    # Keep the projector on the same device/dtype as the language model.
                    model.get_model().mm_projector.to(device=language_model_device, dtype=model.dtype)

                    # Create image_newline parameter if using unpad
                    if 'unpad' in getattr(model.config, 'mm_patch_merge_type', ''):
                        import torch.nn as nn
                        model.get_model().image_newline = nn.Parameter(
                            torch.empty(model.config.hidden_size, dtype=model.dtype)
                        )

                    # Load weights for vision_tower, mm_projector, and image_newline from original checkpoint
                    print("Loading vision_tower, mm_projector, and image_newline weights from checkpoint...")

                    # First, ensure vision_tower is fully loaded (not delay_load)
                    if not model.get_model().vision_tower.is_loaded:
                        print("Loading vision_tower model...")
                        model.get_model().vision_tower.load_model()

                    vision_state_dict = {}
                    projector_state_dict = {}
                    image_newline_weight = None

                    # Load all safetensors files from original model path
                    safetensors_files = sorted(glob.glob(os.path.join(model_path, "model-*.safetensors")))
                    for safetensors_file in safetensors_files:
                        with safe_open(safetensors_file, framework="pt", device="cpu") as f:
                            for key in f.keys():
                                tensor = f.get_tensor(key)

                                # Extract vision_tower weights
                                if key.startswith('vision_tower.'):
                                    # Remove 'vision_tower.' prefix for loading
                                    new_key = key[len('vision_tower.'):]
                                    vision_state_dict[new_key] = tensor

                                # Extract mm_projector weights (stored as multi_modal_projector in checkpoint)
                                elif key.startswith('multi_modal_projector.'):
                                    # Remap: multi_modal_projector.linear_1 -> 0, linear_2 -> 2 (GELU is at index 1)
                                    new_key = key.replace('multi_modal_projector.linear_1', '0').replace('multi_modal_projector.linear_2', '2')
                                    projector_state_dict[new_key] = tensor

                                # Extract image_newline
                                elif key == 'image_newline':
                                    image_newline_weight = tensor

                    # Load vision_tower weights
                    if vision_state_dict:
                        # vision_tower is a CLIPVisionTower, which has a .vision_tower attribute (CLIPVisionModel)
                        # The weights should be loaded into .vision_tower.vision_tower
                        if hasattr(model.get_model().vision_tower, 'vision_tower'):
                            missing_keys, unexpected_keys = model.get_model().vision_tower.vision_tower.load_state_dict(vision_state_dict, strict=False)
                            print(f"Loaded vision_tower weights. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                        else:
                            print("WARNING: vision_tower does not have a vision_tower attribute, skipping weight loading")

                    # Load mm_projector weights
                    if projector_state_dict:
                        missing_keys, unexpected_keys = model.get_model().mm_projector.load_state_dict(projector_state_dict, strict=False)
                        print(f"Loaded mm_projector weights. Missing: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")

                    # Load image_newline weight
                    if image_newline_weight is not None and hasattr(model.get_model(), 'image_newline'):
                        model.get_model().image_newline.data = image_newline_weight.to(device=language_model_device, dtype=model.dtype)
                        print("Loaded image_newline weight")
                else:
                    # llava-v1.5 and other native formats, let from_pretrained handle config
                    model = LlavaLlamaForCausalLM.from_pretrained(
                        model_path,
                        low_cpu_mem_usage=True,
                        tok_keep_ratio=tok_keep_ratio,
                        prune_method=prune_method,
                        selection_strategy=selection_strategy,
                        ablation_mode=ablation_mode,
                        **kwargs
                    )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=model.dtype)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
