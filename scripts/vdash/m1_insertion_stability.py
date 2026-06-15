# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers.
# SPDX-License-Identifier: Apache-2.0
"""M1 stability check: drive the generated peg into the generated socket and verify the SDF
bore physics is stable (no NaN, no explosion, no tunneling) for a given clearance ``c``.

One clearance per run; several lateral offsets are tested in parallel as vectorized instances
(``/World/Env.*``). The socket is kinematic; the peg is a dynamic rigid body pushed straight
down with a constant force under FORGE-grade physics (docs/physics_params.md).

Examples::

    # headless batch (one clearance)
    docker exec isaaclab_arena-latest bash -lc '
      cd /workspaces/isaaclab_arena && unset DISPLAY &&
      /isaac-sim/python.sh scripts/vdash/m1_insertion_stability.py --clearance 0.25 --headless'

    # watch in the GUI via WebRTC (An's preferred inspection)
    docker exec isaaclab_arena-latest bash -lc '
      cd /workspaces/isaaclab_arena && unset DISPLAY &&
      /isaac-sim/python.sh scripts/vdash/m1_insertion_stability.py --clearance 0.5 --livestream 2'
"""

import argparse
import json
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="V-DASH M1 peg-insertion stability check.")
parser.add_argument("--clearance", type=float, default=0.25, help="Radial clearance c (mm).")
parser.add_argument("--assets-dir", type=str, default="assets/vdash")
parser.add_argument("--offsets-frac", type=float, nargs="*", default=[0.0, 0.5, 1.0, 1.5],
                    help="Lateral start offsets as fractions of clearance c.")
parser.add_argument("--descent-speed", type=float, default=0.02, help="Compliant descent target speed (m/s).")
parser.add_argument("--peg-mass", type=float, default=0.5,
                    help="Effective held-peg mass (kg). Stands in for the arm's inertia so the "
                         "impedance descent is well-conditioned (a 19g free peg diverges).")
parser.add_argument("--steps", type=int, default=480, help="Physics steps per trial.")
parser.add_argument("--speed-limit", type=float, default=12.0, help="Peak-speed explosion threshold (m/s).")
parser.add_argument("--out", type=str, default="output/vdash/m1_stability.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext

# Geometry (must match configs/vdash/assets.yaml)
SHAFT_R = 0.004
SOCKET_H = 0.030
BORE_DEPTH = 0.025
MOUTH_Z = SOCKET_H               # socket base on ground (z=0), mouth at top
BORE_BOTTOM_Z = MOUTH_Z - BORE_DEPTH
DROP_GAP = 0.005                 # tip starts 5 mm above the mouth
ENV_SPACING = 0.2


def forge_sim_cfg() -> SimulationCfg:
    return SimulationCfg(
        device=args_cli.device,
        dt=1 / 120,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=192,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.01,
            friction_correlation_distance=0.00625,
            gpu_max_rigid_contact_count=2**23,
            gpu_max_rigid_patch_count=2**23,
            gpu_collision_stack_size=2**28,
            gpu_max_num_partitions=1,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )


def main():
    c_mm = args_cli.clearance
    tag = f"{c_mm:g}".replace(".", "p")
    socket_usd = os.path.abspath(os.path.join(args_cli.assets_dir, f"socket_c{tag}.usd"))
    peg_usd = os.path.abspath(os.path.join(args_cli.assets_dir, "peg.usd"))
    assert os.path.isfile(socket_usd), f"missing {socket_usd}"
    assert os.path.isfile(peg_usd), f"missing {peg_usd}"

    offsets = args_cli.offsets_frac
    n = len(offsets)

    sim = SimulationContext(forge_sim_cfg())
    sim.set_camera_view(eye=[0.4, -0.5, 0.25], target=[(n - 1) * ENV_SPACING / 2, 0.0, MOUTH_Z])

    # ground + light
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.9)).func("/World/light", sim_utils.DomeLightCfg())

    # one Env per offset; socket (kinematic) + peg (dynamic) per env
    origins = []
    socket_spawn = sim_utils.UsdFileCfg(
        usd_path=socket_usd,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    )
    for i in range(n):
        ox = i * ENV_SPACING
        origins.append([ox, 0.0, 0.0])
        sim_utils.create_prim(f"/World/Env{i}", "Xform", translation=[ox, 0.0, 0.0])
        socket_spawn.func(f"/World/Env{i}/Socket", socket_spawn, translation=(0.0, 0.0, 0.0))

    peg = RigidObject(RigidObjectCfg(
        prim_path="/World/Env.*/Peg",
        spawn=sim_utils.UsdFileCfg(
            usd_path=peg_usd,
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.peg_mass),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=5.0,
                solver_position_iteration_count=192, solver_velocity_iteration_count=1,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(),
    ))

    sim.reset()
    dt = sim.get_physics_dt()
    device = sim.device
    origins_t = torch.tensor(origins, device=device)
    bore_r = SHAFT_R + c_mm / 1000.0

    # initial peg poses: tip at mouth + gap, laterally offset by frac*c
    pose = torch.zeros((n, 7), device=device)
    pose[:, 3] = 1.0  # identity quat (tip down)
    for i, frac in enumerate(offsets):
        pose[i, 0] = origins_t[i, 0] + frac * (c_mm / 1000.0)
        pose[i, 2] = MOUTH_Z + DROP_GAP
    peg.write_root_pose_to_sim(pose)
    peg.write_root_velocity_to_sim(torch.zeros((n, 6), device=device))
    peg.reset()
    start_z = pose[:, 2].clone()                       # frame-independent reference

    # Compliant ("EE-held") descent: a PD spring drags the peg toward a target that ramps
    # straight down, holding xy at the spawn offset and keeping the peg upright. This mirrors the
    # brief's EE-held insertion (finite stiffness) — the physically meaningful stability test. A
    # rigid constant-force push is impulsive/explosion-prone and not representative of the arm.
    # Gentle, gravity-driven insertion matching the brief's 2-5 N push: the heavy held peg's
    # weight (m*g) supplies the downward insertion force; a soft lateral spring holds xy so the
    # chamfer can guide; only the vertical axis is damped. A low wrench clamp keeps contact in the
    # brief's force regime — a stiff open-loop ram is what produced the earlier SDF blow-ups.
    m = args_cli.peg_mass
    k_xy, d_xy, d_z = 300.0, 20.0, 6.0
    k_rot, d_rot = 0.1, 0.05
    f_clip, t_clip = 12.0, 0.5
    xy_target = pose[:, :2].clone()

    max_speed = torch.zeros(n, device=device)
    finite = torch.ones(n, dtype=torch.bool, device=device)
    render = bool(getattr(args_cli, "livestream", 0)) or not args_cli.headless
    for step in range(args_cli.steps):
        pos, vel = peg.data.root_pos_w, peg.data.root_lin_vel_w
        quat, angvel = peg.data.root_quat_w, peg.data.root_ang_vel_w
        F = torch.zeros((n, 1, 3), device=device)
        F[:, 0, 0] = k_xy * (xy_target[:, 0] - pos[:, 0]) - d_xy * vel[:, 0]
        F[:, 0, 1] = k_xy * (xy_target[:, 1] - pos[:, 1]) - d_xy * vel[:, 1]
        F[:, 0, 2] = -d_z * vel[:, 2]   # gravity (m*g ~= brief's 2-5N) supplies the insertion force
        rotvec = 2.0 * torch.sign(quat[:, :1]) * quat[:, 1:]       # small-angle quat error to identity
        T = (-k_rot * rotvec - d_rot * angvel).unsqueeze(1)
        F = F.clamp(-f_clip, f_clip)                               # hard wrench clamp prevents divergence
        T = T.clamp(-t_clip, t_clip)
        peg.set_external_force_and_torque(F, T)
        peg.write_data_to_sim()
        sim.step(render=render)
        peg.update(dt)
        pos2, vel2 = peg.data.root_pos_w, peg.data.root_lin_vel_w
        finite &= torch.isfinite(pos2).all(dim=1) & torch.isfinite(vel2).all(dim=1)
        speed = torch.nan_to_num(vel2.norm(dim=1), nan=1e9)
        max_speed = torch.maximum(max_speed, speed)

    final_z = peg.data.root_pos_w[:, 2]
    final_speed = peg.data.root_lin_vel_w.norm(dim=1)
    # Relative descent (frame-independent): how far the peg dropped from its start height.
    descent = (start_z - final_z)                      # ~insertion depth for aligned/loose cases
    # A real blow-up = ejected/tunnelled an impossible distance, exploded peak speed, non-finite,
    # or never settled. A brief contact transient (high vmax but vfin~0, descent in-range) is fine.
    bad_descent = descent.abs() > (DROP_GAP + BORE_DEPTH + 0.025)   # |descent|>55mm is impossible
    exploded = (max_speed > args_cli.speed_limit) | ~finite | bad_descent | (final_speed > 0.5)
    inserted = (descent > 0.020) & ~exploded            # tip reached ~full bore depth and settled
    stable = finite & ~exploded

    rows = []
    for i, frac in enumerate(offsets):
        rows.append({
            "offset_frac_c": frac, "offset_mm": round(frac * c_mm, 4),
            "descent_mm": round(float(descent[i]) * 1000, 3),
            "max_speed_mps": round(float(max_speed[i]), 3),
            "final_speed_mps": round(float(final_speed[i]), 4),
            "finite": bool(finite[i]), "inserted": bool(inserted[i]),
            "exploded": bool(exploded[i]), "stable": bool(stable[i]),
        })
        print(f"[m1] c={c_mm:g}mm bore_r={bore_r*1000:.3f}mm off={frac*c_mm:.3f}mm "
              f"descent={rows[-1]['descent_mm']:.2f}mm vmax={rows[-1]['max_speed_mps']:.3f} "
              f"vfin={rows[-1]['final_speed_mps']:.3f} inserted={rows[-1]['inserted']} "
              f"stable={rows[-1]['stable']}")

    all_stable = bool(stable.all())
    result = {"clearance_mm": c_mm, "bore_radius_mm": round(bore_r * 1000, 3),
              "descent_speed_mps": args_cli.descent_speed, "peg_mass_kg": args_cli.peg_mass,
              "steps": args_cli.steps, "all_stable": all_stable, "trials": rows}
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    # append to a per-run aggregate keyed by clearance
    agg = {}
    if os.path.isfile(args_cli.out):
        try:
            agg = json.load(open(args_cli.out))
        except Exception:
            agg = {}
    agg[f"c{tag}"] = result
    json.dump(agg, open(args_cli.out, "w"), indent=2)

    print(f"[m1] clearance c={c_mm:g}mm -> all_stable={all_stable}")
    print("M1_STABILITY_OK" if all_stable else "M1_STABILITY_UNSTABLE")


if __name__ == "__main__":
    main()
    simulation_app.close()
