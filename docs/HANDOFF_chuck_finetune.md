# B-DASH chuck-loading — fine-tune handoff (training machine)

The recording machine (RTX 5080 / 16 GB) cannot train: GR00T N1.7 is a 3B VLA and
`launch_finetune.py` loads it in fp32 (`load_bf16 = False`, `backbone_trainable_params_fp32 = True`),
so the weights alone are ~12 GB before any activations. Everything up to and including the LeRobot
dataset is DONE and VERIFIED here; this file is what the training machine needs and nothing else.

Branch: `an/bdash-chuck-load`. Full working record: `docs/progress/2026-08-16.md`.

## 1. What to train on

| | path | size |
|---|---|---|
| **train** | `datasets/bdash/chuck_main2000_v2/lerobot` | 2.0 GB |
| **validation** | `datasets/bdash/chuck_pilot250_v2/lerobot` | 233 MB |
| (source HDF5, not needed for training) | `chuck_main2000_v2.hdf5` / `chuck_pilot250_v2.hdf5` | 39 GB / 4.6 GB |

The two sets were recorded with **different seeds** (main seed 1, validation seed 0) on the same
frozen scene, so the validation set is independent and same-distribution. Only the ~2.2 GB of
LeRobot data needs to move; the HDF5s can stay here.

Verified after conversion: 2000 parquet episodes, **186,588 frames** agreeing across `info.json`,
the parquet files and `episodes.jsonl`; 6000 videos (3 views x 2000) with no zero-byte file; no
non-finite value in any `observation.state` or `action`; fps 30.0 recomputed from the timestamps.

## 2. The command

```bash
docker exec -u an isaaclab_arena-cuda_gr00t_gn17 bash -lc '
cd /workspaces/isaaclab_arena && \
DATASET_PATH=datasets/bdash/chuck_main2000_v2/lerobot \
OUTPUT_DIR=models/bdash-gr00t-n1-7-chuck-pick-v1 \
MAX_STEPS=15000 GLOBAL_BATCH_SIZE=16 GRAD_ACCUM=1 \
SAVE_STEPS=1000 SAVE_TOTAL_LIMIT=5 USE_WANDB=false \
bash scripts/bdash/train_vla.sh'
```

`-G` (gn17 = N1.7), **not** `-g` (gn16 = N1.6): `train_vla.sh` says so in its header and the
current best peg model is `umegan/vdash-gr00t-n1-7-pick-handoff-v6-recovery`, i.e. N1.7. Training
N1.6 would not be comparable with anything already measured.

Defaults kept deliberately: `TUNE_LLM=false`, `TUNE_VISUAL=false` (frozen ViT — v5-vision's
unfrozen run regressed), projector + diffusion head trainable.

## 3. Two launcher bugs are fixed on this branch — you need these

`isaaclab_arena_gr00t/scripts/launch_finetune.py` had gone stale against the Isaac-GR00T submodule
and failed before the first step. Both were found by running it, not by reading it:

1. `ft_config.embodiment_tag.value` -> `AttributeError`. `FinetuneConfig.embodiment_tag` is a plain
   `str` in the current submodule; it was an `EmbodimentTag` enum when the launcher was written.
2. With that fixed, `NEW_EMBODIMENT` reached the registry verbatim and matched nothing — the
   registry is keyed by the enum VALUE (`new_embodiment`), which `.value` used to supply.

Both now accept either form, so the peg pipeline still works if the older submodule is checked out.

## 4. What has NOT been verified

**Nothing past model load.** The run here got as far as downloading `nvidia/GR00T-N1.7-3B` and was
stopped; it never reached a training step. So the following are unknown and are yours to find:

* whether the franka modality config matches this dataset's 9-D state / 8-D action end to end
* memory headroom, throughput, and a sensible `MAX_STEPS` for 186,588 frames
* whether the loss curve behaves

## 5. What this dataset is, and the one number worth checking early

2000 demos of "pick a workpiece out of the tray and lift it clear" (the VLA's slice; insertion is
classical control and is not in this data). Per-episode: 3 x 256x256 RGB, 9-D joint state, 8-D
action, mean 94 frames.

* pose mix: **45.8% side-lying** / 54.2% upright, all six variant x pose cells at 13.7-18.9%
* **workpiece appearance is randomised per episode** and is INDEPENDENT of the variant — per-variant
  hue mean 0.485/0.515/0.507 and sd 0.285-0.301 against a uniform draw's 0.500 / 0.289 (n=2000).
  Before this the three variants were fixed colours (blue/orange/green), i.e. the colour was a
  perfect family label; a policy could name the family from one pixel.
* frame freezes: **zero** over 2252 attempts x 3 cameras (this is the M10 bug; it is environment
  specific, so re-check `cam_frozen_steps` on your GPU before trusting a new recording)

The HDF5 also carries `appearance` ([r, g, b, roughness, metallic] of the target), `axis_cond` and
`action_owner` per step. **None are converted into LeRobot** — which channel the policy is
conditioned on binds at conversion, not at recording, so adding one costs a re-conversion (minutes)
and never a re-record.

## 6. The evaluation that answers the appearance claim

`configs/bdash/chuck_load/materials.yaml` defines four **held-out** materials (black oxide, bare
steel, red anodised, brass) at and beyond the edge of the training box. Evaluate the finetuned
policy with `BDASH_HELDOUT_MATERIALS=1`; the gap between the training-material and held-out success
rates is the measurement of "appearance can be a distribution without costing grasp".

Note the A/B slice already run here (randomised vs plain, 30 episodes each) came out
**bit-for-bit identical** and CANNOT answer this: the scripted teacher drives from ground-truth pose
and never looks at a pixel. It does prove the material change did not touch physics.

## 7. Environment note

`isaaclab_arena:cuda_gr00t_gn17` already has everything (torch 2.7.0+cu128, transformers 4.57.3,
gr00t, pyarrow 17.0.0, av 12.3.0, ffmpeg). The conversion here was initially run in the plain
`isaaclab_arena:latest` container, which lacks all of that — do the conversion in gn17 and the
dependency problem does not exist.
