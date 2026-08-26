# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""B-DASH chuck-loading tray scatter and target selection.

Two reset events plus the pure geometry they rest on:

  * :func:`sample_tray_layout` -- non-overlapping resting poses for K workpieces inside the tray
    cavity, a configurable fraction of them lying on their side
  * :func:`scatter_tray_workpieces` -- writes that layout into the sim
  * :func:`select_target_workpiece` -- latches which workpiece the episode is about

**Why not Isaac Lab's** ``sample_object_poses`` **(via** ``randomize_poses_and_align_auxiliary_assets``
**).** It samples roll/pitch/yaw independently and uniformly, so it cannot produce a bimodal
upright/side-lying split; it samples z uniformly rather than deriving it from the orientation; it
checks only centre-to-centre separation, which is meaningless for a 100 mm rod; and on its final try
it accepts the sample *unconditionally*, silently returning interpenetrating poses that blow up
PhysX at reset. This module rejects instead of degrading, and raises if it cannot place a part.

**Poses are placed at rest, not dropped.** Isaac Lab runs the reset event manager and then computes
observations with no physics steps in between, so a drop height would put falling parts in the first
recorded frames and make the teacher chase a moving target. :func:`resting_pose` solves where each
variant actually comes to rest, which is *not* simply "axis horizontal": the stepped variants have
their centre of mass ahead of the fat section, so they pivot down onto the leading end and settle
tilted (W-B by 3.3 deg, W-C by 7.9 deg).

Only ``math``/``random`` are imported at module level -- torch and isaaclab are deferred into the
event functions -- so the sampler can be loaded by file path and tested without a simulator.
"""

from __future__ import annotations

import math
import random

# ======================================================================================
# pure geometry
# ======================================================================================


def resting_pose(sections) -> dict:
    """Where a stepped body of revolution comes to rest lying on a flat floor.

    ``sections`` is ``[(radius, length), ...]`` stacked from the leading end (local z = 0), i.e. the
    form :func:`bdash_chuck_assets.workpiece_profile` returns.

    The part rests on a support line under its radius profile. Writing the axis as height
    ``h(s) = h0 + s*sin(tilt)`` at axial distance ``s``, the lowest surface point of section ``i`` is
    ``h(s) - r_i*cos(tilt)``, so "no section penetrates the floor" is the linear condition
    ``c + m*s >= r_i`` with ``m = tan(tilt)`` and ``c = h0/cos(tilt)``. Among the lines satisfying it
    the part settles on the one minimising the centre-of-mass height. That is a two-variable LP whose
    optimum sits at a vertex, so it is enough to enumerate the lines through pairs of profile corners
    plus the horizontal line at the maximum radius.

    Returns ``tilt_rad`` (positive = the trailing end rides high), ``origin_height`` (the leading end
    face centre, which is the USD prim origin), ``com_s``, ``length`` and ``max_radius``.
    """
    corners, s, volume, moment = [], 0.0, 0.0, 0.0
    for radius, length in sections:
        corners += [(s, radius), (s + length, radius)]
        area = math.pi * radius * radius
        volume += area * length
        moment += area * length * (s + length / 2.0)
        s += length
    total_length, com_s = s, moment / volume
    max_radius = max(r for _, r in corners)

    candidates = [(0.0, max_radius)]  # the flat case: rests on its widest section
    for i in range(len(corners)):
        for j in range(i + 1, len(corners)):
            (s1, r1), (s2, r2) = corners[i], corners[j]
            if abs(s2 - s1) < 1e-12:
                continue
            slope = (r2 - r1) / (s2 - s1)
            candidates.append((slope, r1 - slope * s1))

    best = None
    for slope, intercept in candidates:
        if any(intercept + slope * sp < rp - 1e-12 for sp, rp in corners):
            continue  # this line cuts into the floor somewhere
        tilt = math.atan(slope)
        com_height = math.cos(tilt) * (intercept + slope * com_s)
        if best is None or com_height < best[0]:
            best = (com_height, tilt, math.cos(tilt) * intercept)

    if best is None:  # pragma: no cover - a radius profile always admits its own horizontal line
        raise RuntimeError(f"no resting pose for sections {sections!r}")
    _, tilt_rad, origin_height = best
    return {
        "tilt_rad": tilt_rad,
        "origin_height": origin_height,
        "com_s": com_s,
        "length": total_length,
        "max_radius": max_radius,
    }


def axis_from_tilt(tilt: float, heading: float) -> tuple[float, float, float]:
    """Unit part axis (leading -> trailing) for an elevation ``tilt`` above horizontal at ``heading``."""
    return (math.cos(tilt) * math.cos(heading), math.cos(tilt) * math.sin(heading), math.sin(tilt))


def quat_from_tilt(tilt: float, heading: float) -> tuple[float, float, float, float]:
    """Orientation (w, x, y, z) mapping the workpiece's local +Z onto :func:`axis_from_tilt`.

    ``Rz(heading) * Ry(pi/2 - tilt)``. One construction serves both pose classes -- a side-lying part
    is a small ``tilt`` and an upright one is ``tilt`` near ``pi/2`` -- which is what guarantees the
    quaternion written to the sim and the axis used for collision checking cannot disagree. The
    workpieces are bodies of revolution, so roll about their own axis is unobservable and free.
    """
    cz, sz = math.cos(heading / 2.0), math.sin(heading / 2.0)
    ca, sa = math.cos((math.pi / 2.0 - tilt) / 2.0), math.sin((math.pi / 2.0 - tilt) / 2.0)
    return (cz * ca, -sz * sa, cz * sa, ca * sz)


def _segment_distance_2d(a0, a1, b0, b1) -> float:
    """Shortest distance between two 2-D segments. Degenerate (point) segments are fine."""
    ax, ay = a1[0] - a0[0], a1[1] - a0[1]
    bx, by = b1[0] - b0[0], b1[1] - b0[1]
    wx, wy = a0[0] - b0[0], a0[1] - b0[1]
    aa, bb, cc = ax * ax + ay * ay, ax * bx + ay * by, bx * bx + by * by
    dd, ee = ax * wx + ay * wy, bx * wx + by * wy
    denom = aa * cc - bb * bb

    if denom < 1e-16:  # parallel or degenerate: clamp one parameter and solve the other
        s_num, t_num = 0.0, ee
        s_den, t_den = 1.0, cc if cc > 1e-16 else 1.0
    else:
        s_num, s_den = bb * ee - cc * dd, denom
        t_num, t_den = aa * ee - bb * dd, denom

    s = min(max(s_num / s_den, 0.0), 1.0)
    t = min(max(t_num / t_den, 0.0), 1.0)
    # re-clamp the partner parameter after clamping (standard segment-segment refinement)
    if aa > 1e-16:
        s = min(max((bb * t - dd) / aa, 0.0), 1.0)
    if cc > 1e-16:
        t = min(max((bb * s + ee) / cc, 0.0), 1.0)

    dx = (a0[0] + s * ax) - (b0[0] + t * bx)
    dy = (a0[1] + s * ay) - (b0[1] + t * by)
    return math.hypot(dx, dy)


def jaw_corridor(grasp_xy, axis, side_lying: bool, half_span: float):
    """The horizontal segment the OPEN gripper jaws sweep through when picking at ``grasp_xy``.

    The Franka gripper is binary, so it always descends fully open -- roughly ``half_span`` either
    side of the TCP along the finger-closing direction. Anything sitting in that corridor is struck,
    or worse, quietly gripped instead of the target.

    The closing direction is known per pose class, which is what makes this a thin corridor rather
    than a ``half_span``-radius disc (six parts each needing a 100 mm clear disc do not fit in this
    tray at all): a side-lying part is gripped across its axis, and an upright one is gripped at the
    fixed down-quat, whose fingers close along world y.
    """
    if side_lying:
        norm = math.hypot(axis[0], axis[1]) or 1.0
        close = (-axis[1] / norm, axis[0] / norm)
    else:
        close = (0.0, 1.0)
    return (
        (grasp_xy[0] - half_span * close[0], grasp_xy[1] - half_span * close[1]),
        (grasp_xy[0] + half_span * close[0], grasp_xy[1] + half_span * close[1]),
    )


def piece_top_z(profile: dict, origin_z: float, axis, side_lying: bool) -> float:
    """World height of a placed workpiece's highest point.

    Derived from the live axis rather than a table of per-variant constants, so it keeps working
    once dimensions are sampled continuously per spawn (spec §0-3).
    """
    if side_lying:
        # lying down: the top is the axis height at its highest end, plus that section's radius
        return origin_z + max(0.0, profile["length"] * axis[2]) + profile["max_radius"]
    return origin_z + profile["length"] * abs(axis[2])


def hand_clearance(grasp_xy, grasp_z, close_dir, neighbours, *, half_span, half_width, clear_z):
    """How far the OPEN hand's swept volume is from intruding on a taller neighbour.

    Returns ``(clear, worst_overlap)``. ``clear`` is True when nothing intrudes; ``worst_overlap``
    is the largest vertical intrusion in metres (0.0 when clear) and orders fallback candidates.

    Why this exists: the jaw corridor models only the FINGERS (±50 mm × 12 mm) and is purely 2D. The
    hand BODY hangs its underside just ``clear_z`` above the TCP and spans ``±half_span`` along the
    closing direction — so on a straight-line descent it lands on top of any neighbour taller than
    that underside. Measured, this predicted the observed stall heights to ±2 mm with no free
    parameters (``stall TCP z = neighbour_top − clear_z``).

    ``neighbours`` are ``(xy, top_z, radius)``. The hand is treated as an axis-aligned box in its own
    frame: ``half_span`` along ``close_dir``, ``half_width`` across it.
    """
    hand_bottom = grasp_z + clear_z
    across = (-close_dir[1], close_dir[0])
    worst = 0.0
    for (nx, ny), top_z, radius in neighbours:
        overlap = top_z - hand_bottom
        if overlap <= 0.0:
            continue  # shorter than the hand's underside: it passes over
        dx, dy = nx - grasp_xy[0], ny - grasp_xy[1]
        along = abs(dx * close_dir[0] + dy * close_dir[1])
        side = abs(dx * across[0] + dy * across[1])
        if along <= half_span + radius and side <= half_width + radius:
            worst = max(worst, overlap)
    return worst <= 0.0, worst


def upright_yaw_rule(grasp_xy, hand_bottom_z, neighbours) -> float:
    """Spec §0-11: point the hand's long axis away from the tallest interfering neighbour.

    An upright workpiece is a body of revolution, so the wrist yaw is free — which means it can be
    spent on clearance. The hand is long along the closing direction (±100 mm) and short across it
    (±32 mm), so the clearance-maximising choice is to put the CLOSING direction perpendicular to
    the bearing of the tallest neighbour.

    With ``c(psi) = (sin psi, -cos psi)`` (the convention in ee_control.grasp_quat_from_axis) the
    condition ``c . d = 0`` gives ``psi = atan2(d_y, d_x)``.

    The result is folded to ``[-pi/2, pi/2]``: ``psi`` and ``psi + pi`` describe the same closing
    LINE (the hand is symmetric about its centre), so folding picks one representative
    deterministically and keeps panda_joint7 inside its travel. Determinism matters beyond the
    kinematics — spec §3 requires the rule to be recoverable from the scene alone so the VLA can
    learn it, and a rule with two equally valid answers is not.

    Returns 0.0 when nothing can interfere, which reproduces the plain down-quat exactly.
    """
    tallest = None
    for (nx, ny), top_z, _radius in neighbours:
        if top_z <= hand_bottom_z:
            continue
        if tallest is None or top_z > tallest[1]:
            tallest = ((nx, ny), top_z)
    if tallest is None:
        return 0.0
    (nx, ny), _ = tallest
    dx, dy = nx - grasp_xy[0], ny - grasp_xy[1]
    if abs(dx) < 1e-12 and abs(dy) < 1e-12:
        return 0.0
    psi = math.atan2(dy, dx)
    while psi > math.pi / 2:
        psi -= math.pi
    while psi < -math.pi / 2:
        psi += math.pi
    return psi


def closing_direction(axis, side_lying: bool, upright_yaw: float = 0.0):
    """Unit vector the jaws close along — the hand's long axis.

    Side-lying parts are gripped across their own axis. Upright parts are yaw-symmetric, so the yaw
    is free and chosen by the §0-11 rule; at ``upright_yaw = 0`` this reduces to world -y, which is
    what the fixed down-quat gives.
    """
    if side_lying:
        norm = math.hypot(axis[0], axis[1]) or 1.0
        return (-axis[1] / norm, axis[0] / norm)
    return (math.sin(upright_yaw), -math.cos(upright_yaw))


def _sample_origin(rng, half_x: float, half_y: float, dx: float, dy: float):
    """Uniform point p such that both p and p+(dx,dy) lie in the box [+-half_x, +-half_y].

    Sampling the origin from the yaw-conditioned box rather than rejecting on bounds afterwards is
    what keeps the acceptance rate high for a 100 mm rod in a 190 mm-deep cavity.
    """
    lo_x, hi_x = max(-half_x, -half_x - dx), min(half_x, half_x - dx)
    lo_y, hi_y = max(-half_y, -half_y - dy), min(half_y, half_y - dy)
    if lo_x > hi_x or lo_y > hi_y:
        return None  # this heading does not fit in the cavity at all
    return rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y)


def sample_tray_layout(
    rng: random.Random,
    *,
    pieces: list[dict],
    cavity_xy: tuple[float, float],
    floor_top_z: float,
    side_lying_frac: float = 0.4,
    upright_tilt_deg: float = 3.0,
    min_clearance: float = 0.004,
    wall_inset_body: float = 0.003,
    grasp_wall_margin: float = 0.050,
    grasp_station: float = 0.030,
    place_epsilon: float = 0.001,
    jaw_half_span: float = 0.050,
    jaw_half_width: float = 0.012,
    max_tries: int = 400,
    max_restarts: int = 12,
) -> tuple[list[dict], int]:
    """Non-overlapping resting poses for every piece inside the tray cavity.

    ``pieces`` are :func:`bdash_chuck_assets.workpiece_profile` dicts. Positions are relative to the
    cavity centre in x/y and to the tray root in z, so the caller adds the tray pose and the env
    origin. Returns ``(placements, max_tries_used)``; each placement carries ``pos``, ``quat``
    (w, x, y, z), ``axis`` (the unit part axis, leading -> trailing) and ``side_lying``.

    Raises ``RuntimeError`` rather than returning an overlapping layout -- a loud reset failure is
    far better than a silent PhysX explosion in the middle of a recording run.
    """
    count = len(pieces)

    # An exact COUNT, not a per-piece coin flip: the side-lying fraction then holds in every
    # episode instead of only in expectation, which matters at a 250-demo pilot size.
    side_flags = [False] * count
    for i in rng.sample(range(count), int(round(side_lying_frac * count))):
        side_flags[i] = True

    # Place the biggest footprint first. Greedy sequential placement is prone to painting itself
    # into a corner -- five small parts scattered at random leave nowhere for a side-lying W-C,
    # whose 79 x 45 mm swept capsule is by far the largest -- and largest-first is the standard
    # fix. Ties are broken randomly so which copy of a variant goes down first still varies.
    def _footprint(i: int) -> float:
        piece = pieces[i]
        tilt = resting_pose(piece["sections"])["tilt_rad"] if side_flags[i] else math.pi / 2.0
        span = piece["length"] * math.cos(tilt)
        return span * 2.0 * piece["max_radius"] + math.pi * piece["max_radius"] ** 2

    order = sorted(rng.sample(range(count), count), key=_footprint, reverse=True)

    for _ in range(max_restarts):
        attempt_layout = _place_all(
            rng,
            pieces=pieces,
            side_flags=side_flags,
            order=order,
            cavity_xy=cavity_xy,
            floor_top_z=floor_top_z,
            upright_tilt=math.radians(upright_tilt_deg),
            min_clearance=min_clearance,
            wall_inset_body=wall_inset_body,
            grasp_wall_margin=grasp_wall_margin,
            grasp_station=grasp_station,
            place_epsilon=place_epsilon,
            jaw_half_span=jaw_half_span,
            jaw_half_width=jaw_half_width,
            max_tries=max_tries,
        )
        if attempt_layout is not None:
            return attempt_layout

    raise RuntimeError(
        f"could not lay out {count} workpieces in a {cavity_xy[0]:.3f} x {cavity_xy[1]:.3f} m cavity "
        f"after {max_restarts} restarts of {max_tries} tries -- the cavity, clearances or piece "
        "count are infeasible"
    )


def _place_all(
    rng: random.Random,
    *,
    pieces,
    side_flags,
    order,
    cavity_xy,
    floor_top_z,
    upright_tilt,
    min_clearance,
    wall_inset_body,
    grasp_wall_margin,
    grasp_station,
    place_epsilon,
    jaw_half_span,
    jaw_half_width,
    max_tries,
):
    """One full layout attempt. Returns ``(placements, max_tries_used)``, or None if it jammed."""
    half_x, half_y = cavity_xy[0] / 2.0, cavity_xy[1] / 2.0
    placements: list[dict | None] = [None] * len(pieces)
    segments: list[tuple] = []
    corridors: list[tuple] = []
    worst_tries = 0

    for index in order:
        piece, side = pieces[index], side_flags[index]
        rest = resting_pose(piece["sections"])
        radius = piece["max_radius"]
        body_limit_x = half_x - wall_inset_body - radius
        body_limit_y = half_y - wall_inset_body - radius
        grasp_limit_x = half_x - grasp_wall_margin
        grasp_limit_y = half_y - grasp_wall_margin

        placed = None
        for attempt in range(1, max_tries + 1):
            heading = rng.uniform(-math.pi, math.pi)
            if side:
                # lying down: the solved resting tilt, riding on the leading end and the fat section
                tilt = rest["tilt_rad"]
                origin_z = rest["origin_height"]
            else:
                # standing up: a few degrees off vertical, resting on the rim of the leading face.
                # Safe because every variant tips only past 12.6 deg (COM height vs Ø25 base).
                tilt = math.pi / 2.0 - rng.uniform(0.0, upright_tilt)
                origin_z = piece["lead_radius"] * math.cos(tilt)

            quat = quat_from_tilt(tilt, heading)
            axis = axis_from_tilt(tilt, heading)
            span = piece["length"] * math.cos(tilt)  # horizontal extent of the axis
            dx, dy = span * math.cos(heading), span * math.sin(heading)

            origin = _sample_origin(rng, body_limit_x, body_limit_y, dx, dy)
            if origin is None:
                continue

            grasp_x = origin[0] + grasp_station * axis[0]
            grasp_y = origin[1] + grasp_station * axis[1]
            if abs(grasp_x) > grasp_limit_x or abs(grasp_y) > grasp_limit_y:
                continue  # the open fingers would clip a tray wall on the way down

            far = (origin[0] + dx, origin[1] + dy)
            if any(
                _segment_distance_2d(origin, far, other, other_far) < radius + other_r + min_clearance
                for other, other_far, other_r in segments
            ):
                continue

            # The open jaws must have a clear corridor, in BOTH directions: nothing already placed
            # may sit in this part's corridor, and this part may not sit in anyone else's. Without
            # this, ~79% of grasp stations had a neighbour inside the jaw span -- the gripper propped
            # itself open on the neighbour (measured 44.5 mm between jaws while squeezing a Ø25
            # shaft at 89 N) and the lift then failed. Body clearance alone cannot express this,
            # because the jaws are far wider than the parts.
            corridor = jaw_corridor(
                (origin[0] + grasp_station * axis[0], origin[1] + grasp_station * axis[1]),
                axis,
                side,
                jaw_half_span,
            )
            if any(
                _segment_distance_2d(*corridor, other, other_far) < jaw_half_width + other_r
                for other, other_far, other_r in segments
            ):
                continue
            if any(
                _segment_distance_2d(*other_corridor, origin, far) < jaw_half_width + radius
                for other_corridor in corridors
            ):
                continue

            segments.append((origin, far, radius))
            corridors.append(corridor)
            placed = {
                "pos": (origin[0], origin[1], floor_top_z + origin_z + place_epsilon),
                "quat": quat,
                "axis": axis,
                "side_lying": side,
            }
            worst_tries = max(worst_tries, attempt)
            break

        if placed is None:
            return None  # jammed: the caller restarts the whole layout rather than degrading it
        placements[index] = placed

    return placements, worst_tries  # type: ignore[return-value]


# ======================================================================================
# reset events
# ======================================================================================


def tray_frame(task_cfg: dict) -> tuple[float, float, float]:
    """``(x, y, yaw_rad)`` of the tray in world.

    The tray is YAWED as of 2026-08-16, and the yaw is load-bearing rather than cosmetic. The tray
    is 300 x 200 mm and every task point has to sit in a horizontal annulus 0.35-0.60 m from the
    base (the Franka's usable band): with the long axis RADIAL the grasp region spans 260 mm
    radially against a 250 mm annulus, so no position at any bearing is feasible. Turned tangential
    the radial span drops to 160 mm. The bearings at which a world-axis-aligned tray would already
    be tangential all fall off the table's measured collision footprint, so the rotation is the only
    way the layout closes. See ``scripts/bdash/solve_layout.py``.
    """
    scene = task_cfg["scene"]
    return (
        float(scene["tray_pose"][0]),
        float(scene["tray_pose"][1]),
        math.radians(float(scene.get("tray_yaw_deg", 0.0))),
    )


def tray_to_world(x: float, y: float, frame: tuple[float, float, float]) -> tuple[float, float]:
    """Cavity-frame xy -> world xy. Every consumer of a layout position must go through this."""
    tx, ty, yaw = frame
    c, s = math.cos(yaw), math.sin(yaw)
    return tx + x * c - y * s, ty + x * s + y * c


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """Rotation about world +Z as (w, x, y, z)."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def quat_mul_wxyz(a, b):
    """Hamilton product, (w, x, y, z). Used to yaw a sampled resting pose into the world frame."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def layout_kwargs_from_cfg(task_cfg: dict, assets_cfg: dict, grasp_station: float) -> dict:
    """Assemble :func:`sample_tray_layout` kwargs from the task and asset configs.

    Both the environment's initial spawn and the per-reset scatter go through this, so the layout
    the scene is built with and the layout it is reset to cannot drift apart. Still config-only, so
    it stays importable without a simulator.
    """
    spawn = task_cfg["tray_spawn"]
    tray = assets_cfg["tray"]
    wall = float(tray["wall"])
    return {
        "cavity_xy": (float(tray["outer"][0]) - 2.0 * wall, float(tray["outer"][1]) - 2.0 * wall),
        "floor_top_z": float(task_cfg["scene"]["tray_pose"][2]) + float(tray["floor"]),
        "side_lying_frac": float(spawn["side_lying_frac"]),
        "upright_tilt_deg": float(spawn["upright_tilt_deg"]),
        "min_clearance": float(spawn["min_clearance"]),
        "wall_inset_body": float(spawn["wall_inset_body"]),
        "grasp_wall_margin": float(spawn["grasp_wall_margin"]),
        "grasp_station": float(grasp_station),
        "place_epsilon": float(spawn["place_epsilon"]),
        "jaw_half_span": float(spawn["jaw_half_span"]),
        "jaw_half_width": float(spawn["jaw_half_width"]),
        "max_tries": int(spawn["max_tries"]),
    }


def _layout_rng(env, env_id: int, seed: int) -> random.Random:
    """One persistent RNG per env, so successive resets draw successive layouts reproducibly."""
    store = getattr(env, "_bdash_layout_rngs", None)
    if store is None:
        store = {}
        env._bdash_layout_rngs = store
    if env_id not in store:
        store[env_id] = random.Random((seed * 1_000_003) ^ (env_id * 9_176_083))
    return store[env_id]


def scatter_tray_workpieces(
    env,
    env_ids,
    *,
    asset_names: list[str],
    pieces: list[dict],
    tray_xy: tuple[float, float],
    layout: dict,
    seed: int = 0,
    tray_yaw: float = 0.0,
):
    """Reset event: re-scatter every workpiece to a fresh non-overlapping resting layout.

    ``layout`` comes from :func:`layout_kwargs_from_cfg` and is passed as one dict rather than
    expanded, because Isaac Lab's EventManager introspects the term signature and rejects
    ``**kwargs``. Its ``floor_top_z`` already carries the tray's own height, so only x/y need the
    tray pose added here.

    Also records ``env.bdash_side_lying`` ``(N, K)`` so the target selector can bias toward the
    side-lying parts and the recorder can log which pose class each demo exercised.
    """
    if env_ids is None:
        return
    import torch

    device = env.device
    count = len(asset_names)
    if not hasattr(env, "bdash_side_lying"):
        env.bdash_side_lying = torch.zeros(env.num_envs, count, dtype=torch.bool, device=device)

    for env_id in env_ids.tolist():
        placements, _ = sample_tray_layout(_layout_rng(env, env_id, seed), pieces=pieces, **layout)
        ids = torch.tensor([env_id], device=device)
        origin = env.scene.env_origins[env_id, 0:3]
        frame = (tray_xy[0], tray_xy[1], tray_yaw)
        spin = yaw_quat(tray_yaw)
        for name, place in zip(asset_names, placements):
            x, y, z = place["pos"]
            wx, wy = tray_to_world(x, y, frame)
            # Both halves of the pose must be yawed, not just the position: the sampler solves the
            # resting pose in the CAVITY frame, so a placement rotated in xy while its quaternion
            # stayed in the old frame would put a side-lying part across the tray it was solved to
            # lie along, and the layout's own non-overlap guarantee would no longer describe it.
            pose = torch.tensor([[wx, wy, z, *quat_mul_wxyz(spin, place["quat"])]], device=device)
            pose[:, 0:3] += origin
            asset = env.scene[name]
            asset.write_root_pose_to_sim(pose, env_ids=ids)
            asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=ids)
        env.bdash_side_lying[env_id] = torch.tensor([p["side_lying"] for p in placements], device=device)


def select_target_workpiece(
    env,
    env_ids,
    *,
    num_workpieces: int,
    variant_ids: list[int],
    mode: str = "uniform",
    side_lying_prob: float = 0.5,
    seed: int = 0,
    pieces: list[dict] | None = None,
    asset_names: list[str] | None = None,
    hand: dict | None = None,
    grasp: dict | None = None,
    enforce_hand_clearance: bool = True,
):
    """Reset event: latch which workpiece this episode is about.

    Writes ``env.bdash_target_idx`` and ``env.bdash_target_variant``. Every consumer -- the
    predicates, the scripted teacher, the recorder sidecar -- reads those, so they cannot disagree
    about which of the six parts the episode was scored on.

    ``mode="side_lying_biased"`` forces a side-lying part with probability ``side_lying_prob``,
    which is how the hard-case slice is enriched without changing the tray distribution itself.
    The probability is NOT the achieved rate: on the other branch the §0-10 clearance filter still
    prefers upright parts, so measured with 3 parts, ``side_lying_prob`` 0.5 gave a 75% side-lying
    target rate against 6.7% under ``uniform``. Set it from a measured rate, not from the target.

    **Spec §0-10**: with ``enforce_hand_clearance`` the target is restricted to workpieces the open
    hand can actually descend onto (see :func:`hand_clearance`). The tray still holds six parts —
    this guarantees only that the episode's TARGET is reachable, which is the sampler's existing
    promise ("the open gripper can reach every grasp station") made good, not an easing of the task.
    Choosing WHICH part to work on, and in what order, is ER 2's job at run time.

    Set ``enforce_hand_clearance=False`` for the evaluation slice that reports performance without
    the guarantee (spec §2.2, §3 evaluation set (d)).
    """
    if env_ids is None:
        return
    import torch

    device = env.device
    if not hasattr(env, "bdash_target_idx"):
        env.bdash_target_idx = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        env.bdash_target_variant = torch.zeros(env.num_envs, dtype=torch.long, device=device)

    if not hasattr(env, "bdash_clearance_fallback"):
        env.bdash_clearance_fallback = torch.zeros(env.num_envs, dtype=torch.bool, device=device)

    side_lying = getattr(env, "bdash_side_lying", None)
    for env_id in env_ids.tolist():
        rng = _layout_rng(env, env_id, seed + 1)
        choices = list(range(num_workpieces))
        if mode == "side_lying_biased" and side_lying is not None and rng.random() < side_lying_prob:
            lying = [i for i in choices if bool(side_lying[env_id, i])]
            choices = lying or choices
        elif mode == "side_lying_only" and side_lying is not None:
            # The mirror of `upright_only`, for measuring what the MISSING V-block re-erect (§4-2)
            # actually costs. A side-lying cylinder cannot enter a vertical bore at all, so this
            # mode is expected to load nothing -- the point is to put a number on "nothing" and to
            # see WHERE it stops, since a part that is never even gripped and one that is gripped
            # and then refused by the bore are different problems.
            lying = [i for i in choices if bool(side_lying[env_id, i])]
            choices = lying or choices
        elif mode == "upright_only" and side_lying is not None:
            # spec §4-3 load stage: a side-lying cylinder cannot enter a vertical bore at all, and
            # the V-block re-erect that would fix it (§4-2) is not implemented. Drawing one anyway
            # would fold "did it draw a loadable part" into the load success rate. TEMPORARY --
            # delete this branch when regrasp_vblock lands, so the load stage sees both pose classes.
            standing = [i for i in choices if not bool(side_lying[env_id, i])]
            choices = standing or choices

        fallback = False
        if enforce_hand_clearance and pieces is not None and asset_names is not None:
            clear, ranked = _clearance_partition(env, env_id, choices, pieces, asset_names, side_lying, hand, grasp)
            if clear:
                choices = clear
            elif ranked:
                # Never raise: a scatter that leaves nothing cleanly reachable must still yield an
                # episode, or a long recording run dies on one bad draw. Take the least-obstructed
                # candidate and flag it so the slice can be excluded from the headline number.
                choices = [ranked[0]]
                fallback = True

        pick = rng.choice(choices)
        env.bdash_target_idx[env_id] = pick
        env.bdash_target_variant[env_id] = variant_ids[pick]
        env.bdash_clearance_fallback[env_id] = fallback


def _clearance_partition(env, env_id, choices, pieces, asset_names, side_lying, hand, grasp):
    """Split candidates into (hand-clear, all-ranked-by-obstruction) for one env.

    Poses are read live: ``scatter_tray_workpieces`` has already written them with
    ``write_root_pose_to_sim``, which updates the internal buffers before the PhysX push, so a reset
    event downstream of it sees the scattered layout without stepping physics.
    """
    placed = []
    for k, name in enumerate(asset_names):
        data = env.scene[name].data
        pos = data.root_pos_w[env_id]
        quat = data.root_quat_w[env_id]
        axis = _rotate_z_axis(quat)
        side = bool(side_lying[env_id, k]) if side_lying is not None else False
        top = piece_top_z(pieces[k], float(pos[2]), axis, side)
        placed.append(((float(pos[0]), float(pos[1])), float(pos[2]), axis, side, top, pieces[k]["max_radius"]))

    clear, scored = [], []
    for k in choices:
        (gx0, gy0), pz, axis, side, _top, _r = placed[k]
        below_top = grasp["below_top_upright"]
        if isinstance(below_top, dict):
            below_top = below_top[pieces[k]["variant"]]
        station = grasp["station_side"] if side else pieces[k]["length"] - float(below_top)
        grasp_xy = (gx0 + station * axis[0], gy0 + station * axis[1])
        grasp_z = pz + station * axis[2] + (0.0 if side else 0.0)
        neighbours = [(p[0], p[4], p[5]) for j, p in enumerate(placed) if j != k]
        yaw = 0.0 if side else upright_yaw_rule(grasp_xy, grasp_z + hand["clear_z"], neighbours)
        ok, worst = hand_clearance(
            grasp_xy,
            grasp_z,
            closing_direction(axis, side, yaw),
            neighbours,
            half_span=hand["half_span"],
            half_width=hand["half_width"],
            clear_z=hand["clear_z"],
        )
        scored.append((worst, k))
        if ok:
            clear.append(k)
    scored.sort()
    return clear, [k for _w, k in scored]


def _rotate_z_axis(quat) -> tuple[float, float, float]:
    """Local +Z of a (w,x,y,z) quaternion, as plain floats."""
    w, x, y, z = (float(v) for v in quat)
    return (2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))
