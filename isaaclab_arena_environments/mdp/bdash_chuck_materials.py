# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-episode workpiece appearance, and the fixed materials for everything else.

Isaac Lab ships no visual-randomisation event in this version (``isaaclab.envs.mdp.events`` covers
mass, scale, COM, gravity and pose but nothing renderable), so the reset event here writes the
``UsdPreviewSurface`` shader inputs directly.

**Hue is sampled independently of the variant, and that is the point.** Before this module the
three variants were flat constant colours -- W-A blue, W-B orange, W-C green -- so the colour was a
perfect family label. A policy could name the family from a single pixel without ever looking at
the geometry, which makes the §0-3 claim (continuous dimensions carry the language-selection task)
impossible to test: the shortcut would always win. Sampling hue per part per episode removes it.

**Only appearance.** Nothing here touches a physics material, a collider or a mass, so every
measurement taken before this module existed is unaffected.

Sim-free except for the event functions themselves: the colour maths is plain Python so it can be
unit-tested, and ``pxr`` is imported inside the event.
"""

from __future__ import annotations

import colorsys
import random


def sample_appearance(rng: random.Random, cfg: dict) -> dict:
    """One draw from the training distribution: ``{rgb, roughness, metallic, family}``.

    Draws a material FAMILY first (wood / metal) and then samples inside it, because the two occupy
    very different corners of the space and a single box spanning both would spend most of its mass
    on the semi-saturated semi-metallic middle that is neither. Family is drawn independently of the
    variant, exactly as hue is -- if it tracked the variant it would be the old colour-as-label
    defect again.

    HSV rather than independent RGB channels: sampling RGB uniformly concentrates draws around
    mid-grey and makes saturated colours rare, which is precisely where wood lives.

    Falls back to the flat top-level ranges when no ``families`` block is present.
    """
    families = cfg.get("families")
    if families:
        names = sorted(families)
        weights = [float(families[n].get("weight", 1.0)) for n in names]
        family = rng.choices(names, weights=weights, k=1)[0]
        box = families[family]
    else:
        family, box = "flat", cfg
    hue = rng.uniform(*box["hue"])
    sat = rng.uniform(*box["saturation"])
    val = rng.uniform(*box["value"])
    return {
        "rgb": tuple(colorsys.hsv_to_rgb(hue, sat, val)),
        "roughness": rng.uniform(*box["roughness"]),
        "metallic": rng.uniform(*box["metallic"]),
        "family": family,
    }


def _appearance_rng(env, env_id: int, seed: int) -> random.Random:
    """One PERSISTENT RNG per env, so successive resets draw successive appearances.

    Persistent, not rebuilt per reset. Constructing ``random.Random(f(seed, env_id))`` inside the
    reset event -- which is what this did first -- reseeds to the same state every episode, so every
    episode gets the SAME three appearances and the randomisation is a no-op that looks like it
    works. It survived a render check too: the parts are re-scattered each reset, so their rendered
    colour moves with the shading even when the material never changes, and the measured
    "colour spread across resets" was pose-dependent shading rather than material.

    ``bdash_chuck_randomization._layout_rng`` already had exactly this shape for the same reason.
    """
    store = getattr(env, "_bdash_appearance_rngs", None)
    if store is None:
        store = {}
        env._bdash_appearance_rngs = store
    if env_id not in store:
        store[env_id] = random.Random((seed * 7_919 + 13) ^ (env_id * 2_654_435_761))
    return store[env_id]


def _shader_at(stage, prim_path: str):
    """The surface shader of the material actually BOUND under ``prim_path``, or None.

    Resolved through ``UsdShade.MaterialBindingAPI``, not by walking for the first ``Shader`` prim.
    A workpiece carries TWO materials::

        .../bdash_workpiece_wa_0/geometry/Looks/DefaultMaterial/DefaultMaterial   (from the USD)
        .../bdash_workpiece_wa_0/material/Shader                                  (bound, rendered)

    and "first Shader found" picks the first by path order -- ``geometry`` sorts before
    ``material`` -- so it wrote to the one nothing renders. The write SUCCEEDS, so nothing errors:
    measured, every part kept its spawn-time placeholder colour (W-A rendered (48,122,190), exactly
    the (0.05,0.25,0.95) placeholder) through an entire randomised recording run.
    """
    from pxr import UsdShade

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    # The MESH's binding first: a binding authored on the mesh overrides one inherited from the
    # root, so asking the root can name a material the mesh does not actually use.
    candidates = [child for child in stage.Traverse() if str(child.GetPath()).startswith(prim_path + "/")]
    for candidate in [c for c in candidates if c.IsA(UsdShade.Material) is False and c.GetTypeName() == "Mesh"] + [
        prim
    ]:
        material = UsdShade.MaterialBindingAPI(candidate).ComputeBoundMaterial()[0]
        if not material:
            continue
        shader = material.ComputeSurfaceSource()[0]
        if shader:
            return shader
    return None


def _write(shader, appearance: dict) -> bool:
    """Set diffuse/roughness/metallic on a UsdPreviewSurface shader. True if anything was written."""
    from pxr import Gf, Sdf

    if shader is None:
        return False
    written = False
    for name, value, sdf_type in (
        ("diffuseColor", Gf.Vec3f(*[float(c) for c in appearance["rgb"]]), Sdf.ValueTypeNames.Color3f),
        ("roughness", float(appearance["roughness"]), Sdf.ValueTypeNames.Float),
        ("metallic", float(appearance["metallic"]), Sdf.ValueTypeNames.Float),
    ):
        if value is None:
            continue
        shader.CreateInput(name, sdf_type).Set(value)
        written = True
    return written


def _mdl_library(env, cfg: dict) -> list[tuple[str, str, str]]:
    """Spawn every configured MDL ONCE under /World/Looks and return (family, name, prim_path).

    Spawned once and re-BOUND per episode, never re-spawned: an MDL carries texture maps that have
    to be fetched and compiled, so creating them per reset would stall every episode on I/O. Binding
    is a cheap USD relationship edit.
    """
    from isaaclab.sim.spawners.materials import MdlFileCfg
    from isaaclab.utils.assets import NVIDIA_NUCLEUS_DIR

    cached = getattr(env, "_bdash_mdl_library", None)
    if cached is not None:
        return cached

    root = str(cfg["root"]).replace("{NVIDIA_NUCLEUS_DIR}", NVIDIA_NUCLEUS_DIR)
    entries: list[tuple[str, str, str]] = []
    wanted: list[tuple[str, str]] = []
    for family, box in sorted(cfg.get("families", {}).items()):
        wanted += [(family, rel) for rel in box.get("materials", [])]
    wanted += [("held_out", rel) for rel in cfg.get("held_out", [])]

    for family, rel in wanted:
        name = rel.rsplit("/", 1)[-1].replace(".mdl", "")
        prim_path = f"/World/Looks/bdash_{name}"
        MdlFileCfg(mdl_path=f"{root}/{rel}", project_uvw=True).func(
            prim_path, MdlFileCfg(mdl_path=f"{root}/{rel}", project_uvw=True)
        )
        entries.append((family, name, prim_path))

    env._bdash_mdl_library = entries
    return entries


def _bind_mdl(stage, prim_path: str, material_path: str) -> bool:
    """Bind an already-spawned MDL material to ``prim_path`` and every mesh under it.

    The ROOT prim is bound, and bound STRONGER THAN DESCENDANTS, because that is what the spawner
    did: `visual_material_path` authors a UsdPreviewSurface at `<prim>/material` and binds it on the
    root. Binding only the meshes therefore changed nothing that renders -- measured, the parts kept
    the spawn-time placeholder colour (W-A stayed at its (66,141,200) blue) across four different
    MDL draws, with no error anywhere, because the root binding still won.
    """
    from pxr import UsdShade

    material = UsdShade.Material(stage.GetPrimAtPath(material_path))
    if not material:
        return False
    bound = False
    root = stage.GetPrimAtPath(prim_path)
    if root and root.IsValid():
        api = UsdShade.MaterialBindingAPI.Apply(root)
        api.Bind(material, UsdShade.Tokens.strongerThanDescendants)
        bound = True
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path.startswith(prim_path + "/") and prim.GetTypeName() == "Mesh":
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
            bound = True
    return bound


def randomize_workpiece_appearance(
    env,
    env_ids,
    *,
    asset_names: list[str],
    cfg: dict,
    seed: int = 0,
    held_out: bool = False,
):
    """Reset event: redraw every workpiece's visual material.

    ``held_out=True`` cycles the evaluation materials instead of sampling, so a generalisation
    check runs on appearances the policy has never been trained on.

    Records ``env.bdash_appearance`` -- a list per env of the drawn ``{rgb, roughness, metallic}``
    -- so the recorder can log what each demo actually looked like. Without that the dataset says
    only "randomised" and no failure can ever be attributed to an appearance.
    """
    if env_ids is None or not cfg.get("enabled", True):
        return
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if not hasattr(env, "bdash_appearance"):
        env.bdash_appearance = [None] * env.num_envs

    # MDL PATH: bind a real NVIDIA material (grain, brush direction, normal maps) instead of writing
    # flat PreviewSurface numbers. Falls back to the parameter path if disabled or unavailable.
    mdl_cfg = cfg.get("mdl") or {}
    if mdl_cfg.get("enabled"):
        library = _mdl_library(env, mdl_cfg)
        choices = (
            [e for e in library if e[0] != "held_out"] if not held_out else [e for e in library if e[0] == "held_out"]
        )
        if not choices:
            raise RuntimeError("appearance randomisation: the MDL library is empty; nothing to bind")
        weights = {f: float(b.get("weight", 1.0)) for f, b in (mdl_cfg.get("families") or {}).items()}
        for env_id in env_ids.tolist():
            rng = _appearance_rng(env, env_id, seed)
            drawn = []
            # WITHOUT REPLACEMENT within an episode. The instruction now names the target by its
            # material ("pick up the brushed bronze workpiece"), so two parts sharing one material
            # make the sentence ambiguous and the demo unlearnable -- the same words, two valid
            # answers. Drawing independently, that happens 23.6% of the time across the 12-material
            # library, and 44.4% of the time when all three land in one family.
            #
            # This does NOT reintroduce colour-as-label: the draw is still independent of the
            # VARIANT, so which material a W-B gets is still uncorrelated with it being a W-B. All
            # that changes is that the three parts present at once are distinguishable, which is
            # what makes naming one of them a well-posed instruction.
            taken: set[str] = set()
            for name in asset_names:
                pool = [e for e in choices if e[1] not in taken] or choices
                entry = rng.choices(pool, weights=[weights.get(e[0], 1.0) for e in pool], k=1)[0]
                taken.add(entry[1])
                prim_path = env.scene[name].cfg.prim_path.replace("env_.*", f"env_{env_id}")
                if not _bind_mdl(stage, prim_path, entry[2]):
                    raise RuntimeError(
                        f"appearance randomisation could not bind {entry[2]!r} under {prim_path!r}; "
                        "refusing to record a dataset whose materials are silently constant"
                    )
                drawn.append({"family": entry[0], "name": entry[1], "mdl": entry[2]})
            env.bdash_appearance[env_id] = drawn
        return

    pool = cfg.get("held_out") or []
    for env_id in env_ids.tolist():
        rng = _appearance_rng(env, env_id, seed)
        drawn = []
        for index, name in enumerate(asset_names):
            if held_out and pool:
                entry = pool[(env_id + index) % len(pool)]
                appearance = {
                    "rgb": tuple(float(c) for c in entry["rgb"]),
                    "roughness": float(entry["roughness"]),
                    "metallic": float(entry["metallic"]),
                    "name": entry.get("name", "held_out"),
                }
            else:
                appearance = sample_appearance(rng, cfg)
            prim_path = env.scene[name].cfg.prim_path.replace("env_.*", f"env_{env_id}")
            # LOUD on failure. A silent no-op here is what produced a whole recording run in which
            # every part kept its spawn-time placeholder colour -- i.e. the colour was still a
            # perfect variant label, which is the exact defect this event exists to remove. There
            # is no safe way to continue: the dataset would be mislabelled as randomised.
            if not _write(_shader_at(stage, prim_path), appearance):
                raise RuntimeError(
                    f"appearance randomisation found no bound surface shader under {prim_path!r}; "
                    "refusing to record a dataset whose materials are silently constant"
                )
            drawn.append(appearance)
        env.bdash_appearance[env_id] = drawn


def preview_surface_kwargs(entry: dict) -> dict:
    """``PreviewSurfaceCfg`` kwargs for a fixed fixture material from materials.yaml."""
    return {
        "diffuse_color": tuple(float(c) for c in entry["rgb"]),
        "roughness": float(entry["roughness"]),
        "metallic": float(entry["metallic"]),
    }
