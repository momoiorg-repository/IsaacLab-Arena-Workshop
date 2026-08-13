# B-DASH VLA front-end interface (FROZEN — brief v3 trigger)

Status: **frozen at M9**. This document is the contract for the VLA front-end. VLA fine-tuning is out
of scope here and is carried out under **brief v3 (separate repo)**; M9 completion triggers v3.

## Role of the VLA

In the B-DASH hierarchy the VLA owns **perception + transport**: from the initial scene it must drive
the arm to **pick the peg and bring it to the handoff pose** over the socket. At handoff, control is
passed to the rule-based insertion controller (which the VLA never sees). So the VLA's learning
target is the **pick → handoff** segment only; insertion is not part of this interface.

During development the VLA is stood in for by the privileged scripted expert (`bdash_scripted`); the
M9 demos are recorded from that expert (brief §2.1 permits the scripted expert as the VLA's teacher).

## Handoff contract (where the VLA hands off)

The boundary is the §3.4 **`handoff` predicate** (`isaaclab_arena_environments/mdp/bdash_peg_predicates.py`):

> `grasped` ∧ peg tip inside the handoff cylinder (radius `r_h = 15 mm`, height `20–60 mm` above the
> socket mouth) ∧ peg speed `< 0.05 m/s`.

The VLA episode is considered complete when this predicate first holds. The M6 switcher uses the same
predicate to transfer control to the insertion controller.

## Observations (VLA inputs)

| Key | Shape | Notes |
|---|---|---|
| `wrist_cam_rgb` | 256×256×3 uint8 | wrist camera (`panda_hand/wrist_cam`) |
| `left_cam_rgb` | 256×256×3 uint8 | external camera (left of workspace) |
| `right_cam_rgb` | 256×256×3 uint8 | external camera (right of workspace) |
| `robot_joint_pos` | 9 | 7 arm joints + 2 finger joints |
| `eef_pos` / `eef_quat` | 3 / 4 | end-effector pose (`ee_frame`) |
| `gripper_pos` | 1 | gripper opening |

Cameras come from `FrankaCameraCfg` (Arena Franka embodiment, RGB only). The M8 overhead camera and
the simulator's ground-truth object poses are **not** part of the VLA interface. The minimal image
set for v3 is **external + wrist RGB** (the brief's spec); the two external views are `left/right_cam`.

## Action space

7-D **relative-IK pose delta + binary gripper** (the Franka `DifferentialInverseKinematicsAction`,
`command_type='pose'`, relative mode, `scale=0.5`):

```
[ dx, dy, dz, ax, ay, az, gripper ]   # 6-D world-frame pose delta (axis-angle) + gripper
```

`gripper < 0` closes, `>= 0` opens. Applied at the env control rate (physics dt = 1/120 s,
decimation 2 → **60 Hz**). This is exactly what the scripted expert emits and what the demos store.

**Action-chunk target (v3):** GR00T-N1.6-style **16-step action horizon** (pad the 7-D action to the
8-D Franka action head used by `isaaclab_arena_gr00t`). The recorded demos are per-step 7-D actions;
chunking into 16×8 is done at v3 training time, not at record time.

## Language instruction

Template (parameterizable in v3 for instruction diversity):

> "Pick up the peg and move it over the socket."

Distractor variants for L3 (v3): "Pick up the **blue** peg …" etc.

## Dataset

- **Recording** (M9): `scripts/bdash/record_vla_demos.py` drives `bdash_scripted`, records
  `(obs, action)` per step from reset until the `handoff` predicate fires, exports **successful only**
  to an Isaac Lab **HDF5** dataset. L1-centric; single-env (the non-tiled cameras render one env).
- **LeRobot conversion**: `isaaclab_arena/scripts/imitation_learning/convert_franka_to_lerobot.py`
  → `observation.images.ego_view` (mp4) + parquet (state/action), `robot_type: franka`.
- **Scale**: brief calls for several hundred episodes (L1-centric). Run the recorder in the
  background to accumulate; resume by appending to the dataset directory.

```bash
unset DISPLAY
# 1) record (repeat / background to reach hundreds)
/isaac-sim/python.sh scripts/bdash/record_vla_demos.py --enable_cameras --num_envs 1 --seed 0 \
    --num_demos 200 --dataset_file datasets/bdash/vla_pick_handoff.hdf5 \
    bdash_pick_insert --clearance 2.0 --level L1
# 2) convert to LeRobot
/isaac-sim/python.sh isaaclab_arena/scripts/imitation_learning/convert_franka_to_lerobot.py \
    --input datasets/bdash/vla_pick_handoff.hdf5 --output datasets/bdash/vla_pick_handoff_lerobot
```

## Frozen — change control
Any change to the observation keys/shapes, action space, control rate, or handoff predicate after M9
must be versioned here and communicated to the v3 repo, since the VLA is trained against this
contract.
