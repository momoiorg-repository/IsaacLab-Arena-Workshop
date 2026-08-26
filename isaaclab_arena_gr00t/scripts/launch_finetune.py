# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Custom launch script properly integrated into the repo
import os
import sys
from pathlib import Path

import tyro

# Add submodule to path so we can import from gr00t
SUBMODULE_PATH = Path(__file__).resolve().parents[2] / "submodules" / "Isaac-GR00T"
if str(SUBMODULE_PATH) not in sys.path:
    sys.path.append(str(SUBMODULE_PATH))

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run as run_experiment


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib

    path = Path(modality_config_path).resolve()
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"

    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description="Launch finetuning for GR00T (Fixed for Franka/Memory)")
    # `FinetuneConfig.embodiment_tag` is a plain `str` in the current Isaac-GR00T submodule but was
    # an `EmbodimentTag` enum in the version this launcher was written against, so the unconditional
    # `.value` raised AttributeError on every run. Accept both rather than pinning to one: the peg
    # pipeline's results are frozen against the older submodule and must keep working if it is ever
    # checked back out.
    embodiment_tag = getattr(ft_config.embodiment_tag, "value", ft_config.embodiment_tag)
    # ...and a bare enum NAME still has to become its VALUE. The registry is keyed by the value
    # ("new_embodiment"); the enum did that conversion implicitly via `.value`, so with a plain str
    # the CLI's "NEW_EMBODIMENT" reached the registry verbatim and no tag matched.
    from gr00t.data.embodiment_tags import EmbodimentTag

    if isinstance(embodiment_tag, str) and embodiment_tag in EmbodimentTag.__members__:
        embodiment_tag = EmbodimentTag[embodiment_tag].value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    config = get_default_config().load_dict({
        "data": {
            "download_cache": False,
            "datasets": [{
                "dataset_paths": [ft_config.dataset_path],
                "mix_ratio": 1.0,
                "embodiment_tag": embodiment_tag,
            }],
        }
    })
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params

    # Standard settings
    config.model.load_bf16 = False  # Explicitly False as requested
    config.model.reproject_vision = False
    # N1.7: the model defaults to Gr00tN1d7Config (Cosmos-Reason2-2B / Qwen3-VL backbone). Do NOT set
    # the N1.6 Eagle backbone (model_name) or eagle_collator (the field no longer exists) — let the
    # Gr00tN1d7Config defaults apply; the backbone weights come from start_from_checkpoint (N1.7-3B).
    config.model.backbone_trainable_params_fp32 = True
    config.model.use_relative_action = True

    config.training.start_from_checkpoint = ft_config.base_model_path
    # BDASH_OPTIM env knob: adamw_torch fits frozen-vision on A40 (42.3/46GB measured);
    # visual-tuning runs OOM with it and need adamw_bnb_8bit (vdi00035-002 field data).
    config.training.optim = os.environ.get("BDASH_OPTIM", "adamw_torch")
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = "finetune-gr00t-n1d7"

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

    run_experiment(config)
