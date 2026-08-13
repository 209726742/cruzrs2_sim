#!/usr/bin/env python3
"""factory_assets.py — shared injection of the 3 Tripo factory props into
assets/cruzr_newusd_scene.xml (used by place_assets.py AND cruzr_teleop.py /
grasp_smoke.py, so placement mode and teleop see the exact same assets).

Assets (built by scripts/process_factory_assets.py):
  assets/factory/manifest.json           part list per slug (decimated OBJ + PNG)
  assets/factory/<slug>/proc/vis_*.obj   height-normalized (height=1.0, xy-centred,
                                         bottom at z=0) decimated visual meshes
Pose source:
  assets/factory/placement.json          written by place_assets.py, hand-editable:
    { "<slug>": { "pos":  [x, y, z],        # world position of the asset origin
                                            #  (origin = footprint centre at the
                                            #   BOTTOM face -> z=0 rests on floor)
                  "quat": [w, x, y, z],     # world orientation
                  "scale": s | [sx,sy,sz] } }
                                            # scale is EITHER a scalar (uniform, real
                                            #  height in metres since mesh height=1)
                                            #  OR a per-axis 3-vector [sx,sy,sz] to hit a
                                            #  non-square target world AABB:
                                            #  sx = target_world_X / native_mesh_footprint_X
                                            #  sy = target_world_Y / native_mesh_footprint_Y
                                            #  sz = target_world_Z (= real height, mesh h=1)
                                            #  (identity quat -> mesh axes map to world axes)
  Missing file / missing slug -> built-in DEFAULTS (open floor, clear of the
  rack->table transport corridor x∈[0.3,1.4] y∈[-7,0.4] and of the base-drive
  reverse path x∈[-1.8,0.3] y∈[-0.6,0.6]).

Collision: NOT the raw mesh — each visual part is duplicated as a group-3 mesh
geom, which MuJoCo collides via its convex hull (static furniture => coarse
per-part hulls are exactly what we want; robot/tote cannot pass through).
"""
import json, os
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets"))
SCENE_XML = os.path.join(ASSETS, "cruzr_newusd_scene.xml")
FACTORY = os.path.join(ASSETS, "factory")
MANIFEST = os.path.join(FACTORY, "manifest.json")
PLACEMENT_JSON = os.path.join(FACTORY, "placement.json")
FACTORY_V2 = os.path.join(ASSETS, "factory_v2")
MANIFEST_V2 = os.path.join(FACTORY_V2, "manifest.json")

# jig_base is NO LONGER an injected static prop — it is the MANIPULAND, authored
# directly in cruzr_newusd_scene.xml (body "jig", replaces the old tote). Only the
# two static furniture props are injected here.
SLUGS = ["machinery", "server_rack"]

# default poses (2026-07-07 v3 layout, synced to outputs/second.usd via the
# second-USD->MuJoCo frame map  X'=Y+0.45, Y'=-X, Z'=Z ; identity quat because both
# targets are axis-aligned in world).  Per-axis scale = target_world_dim / native_mesh
# footprint (machinery native [0.6957,2.1329,1], server_rack native [0.3813,0.6879,1]).
#  machinery   : static, lying horizontal, long axis along MuJoCo Y; bottom-centre
#                (0.627,-1.281,0); world AABB 0.671(x) x 2.013(y) x 0.861(z).
#  server_rack : static, UPRIGHT on its casters, NEAR the robot (replaces the far
#                goal). The Tripo mesh's true "up" axis is native +Y (all 4 casters sit
#                on the native -Y face at the X-Z base corners), NOT native +Z, so the
#                identity-quat placement rendered TIPPED ON ITS SIDE. Fixed 2026-07-07:
#                rotate +90deg about world X (quat [0.7071,0.7071,0,0]) so native +Y ->
#                world +Z; casters land on the floor. Per-axis scale is applied in mesh
#                frame BEFORE the rotation, so it maps as:
#                  world height Z = mesh-Y-extent(0.6879) * sy=1.939 -> 1.334 m
#                  world width  X = mesh-X-extent(0.3813) * sx=1.994 -> 0.760 m
#                  world depth  Y = mesh-Z-extent(1.0)    * sz=1.939 -> 1.939 m
#                These dimensions come from outputs/second.usd /World/server_rack_3d_model
#                mapped by X'=Y+0.45, Y'=-X, Z'=Z.
#                pos.z=0.676 matches the USD centre; pos.y is the near edge because native
#                +Z maps to MuJoCo -Y after the +90deg X rotation.
#                (the rotation shifts the mesh's asymmetric Z span into -Y).
#                Final world AABB x[-2.396,-1.636] y[-2.114,-0.175] z[0.009,1.343];
#                flat top place-target at world z~1.300, clear of the base-drive path.
DEFAULTS = {
    "machinery":   {"pos": [ 0.627, -1.281, 0.0], "quat": [1, 0, 0, 0],
                    "scale": [0.964496, 0.943785, 0.861]},
    "server_rack": {"pos": [-2.0163, -0.1747, 0.6763], "quat": [0.70710678, 0.70710678, 0, 0],
                    "scale": [1.993632, 1.939258, 1.939300]},
}

def load_placement():
    """DEFAULTS overlaid with assets/factory/placement.json (if present)."""
    place = {k: dict(v) for k, v in DEFAULTS.items()}
    if os.path.exists(PLACEMENT_JSON):
        try:
            user = json.load(open(PLACEMENT_JSON))
            for slug, p in user.items():
                if slug in place:
                    place[slug].update(p)
        except Exception as e:  # malformed file -> defaults, but say so
            print(f"[factory_assets] WARNING: bad {PLACEMENT_JSON}: {e} -> using defaults")
    return place


def save_placement(place):
    json.dump(place, open(PLACEMENT_JSON, "w"), indent=1)
    print(f"[factory_assets] saved {PLACEMENT_JSON}")


def inject(spec, free=False, placement=None, task_rack_slot=False):
    """Add the 3 factory props to an MjSpec of the scene.
    free=True  -> each asset body gets a (damped) freejoint: placement mode.
    free=False -> static bodies welded to world: teleop / tests."""
    if not os.path.exists(MANIFEST):
        print(f"[factory_assets] {MANIFEST} missing -> scene without factory props")
        return spec
    manifest = json.load(open(MANIFEST))
    place = placement or load_placement()
    for slug in SLUGS:
        if slug not in manifest:
            continue
        p = place[slug]
        sc = p["scale"]
        s = [float(sc)] * 3 if isinstance(sc, (int, float)) else [float(v) for v in sc]
        body = spec.worldbody.add_body(name=f"fa_{slug}", pos=p["pos"], quat=p["quat"])
        if free:
            jnt = body.add_freejoint()
            jnt.name = f"fa_{slug}_free"
            jnt.damping = [20.0, 0, 0]  # so dragged bodies stop when released (g=0)
        for i, part in enumerate(manifest[slug]["parts"]):
            tex = spec.add_texture(name=f"fa_{slug}_tex{i}",
                                   type=mujoco.mjtTexture.mjTEXTURE_2D,
                                   file=os.path.join(FACTORY, part["png"]))
            mat = spec.add_material(name=f"fa_{slug}_mat{i}", specular=0.1, shininess=0.2)
            mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
            mesh = spec.add_mesh(name=f"fa_{slug}_m{i}",
                                 file=os.path.join(FACTORY, slug, "proc", os.path.basename(part["obj"])),
                                 scale=s)
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
            # visual: textured, no collision, massless
            body.add_geom(name=f"fa_{slug}_vis{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=mesh.name, material=mat.name,
                          contype=0, conaffinity=0, group=1, density=0)
            # Part 0 is the complete bank of visually open bins. Its single convex
            # hull seals every opening, so the automatic insertion task replaces
            # only that collider with a measured open target bin below.
            if task_rack_slot and slug == "server_rack" and i == 0:
                continue
            # collision: convex hull of the same mesh (group 3 = hidden, like the
            # rest of this scene's colliders); static furniture contact recipe
            body.add_geom(name=f"fa_{slug}_col{i}", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname=mesh.name, group=3, rgba=[0.5, 0.5, 0.5, 1],
                          condim=3, friction=[1.0, 0.01, 0.001])
    if task_rack_slot:
        collision = dict(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            group=3,
            rgba=[0.5, 0.5, 0.5, 1],
            condim=3,
            friction=[1.0, 0.01, 0.001],
        )
        spec.worldbody.add_geom(name="fa_task_bin_floor", pos=[-1.84, -0.65, 0.887],
                                size=[0.180, 0.25, 0.010], **collision)
        spec.worldbody.add_geom(name="fa_task_bin_back", pos=[-1.84, -0.90, 0.955],
                                size=[0.180, 0.010, 0.068], **collision)
        spec.worldbody.add_geom(name="fa_task_bin_left", pos=[-2.03, -0.65, 0.955],
                                size=[0.010, 0.25, 0.068], **collision)
        spec.worldbody.add_geom(name="fa_task_bin_right", pos=[-1.65, -0.65, 0.955],
                                size=[0.010, 0.25, 0.068], **collision)
    return spec


V2_STATIC_SLUGS = (
    "checkout_counter",
    "modular_server_drive_rack",
    "precision_machine_part",
    "kitchen_base_cabinet",
)


def _box_from_limits(name, body, lo, hi):
    lo = [float(value) for value in lo]
    hi = [float(value) for value in hi]
    body.add_geom(
        name=name,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        pos=[(lo[i] + hi[i]) * 0.5 for i in range(3)],
        size=[(hi[i] - lo[i]) * 0.5 for i in range(3)],
        group=3,
        rgba=[0.5, 0.5, 0.5, 1.0],
        condim=3,
        friction=[1.0, 0.01, 0.001],
    )


def _add_checkout_collision(body, lo, hi, prefix):
    z0, z1 = lo[2], hi[2]
    _box_from_limits(f"{prefix}0", body, [lo[0], lo[1], z1 - 0.12], hi)
    _box_from_limits(f"{prefix}1", body, lo, [hi[0], hi[1], z0 + 0.10])
    _box_from_limits(
        f"{prefix}2", body,
        [lo[0] + 0.025, lo[1] + 0.025, z0 + 0.10],
        [hi[0] - 0.025, hi[1] - 0.025, z1 - 0.12],
    )


def _add_cabinet_collision(body, lo, hi, prefix):
    _box_from_limits(f"{prefix}0", body, lo, [hi[0], hi[1], lo[2] + 0.06])
    _box_from_limits(f"{prefix}1", body, [lo[0], lo[1], lo[2] + 0.06], hi)


def _add_open_frame_collision(body, lo, hi, prefix, shelves=()):
    wall = min(0.018, (hi[0] - lo[0]) * 0.08)
    deck = min(0.014, (hi[2] - lo[2]) * 0.09)
    back = min(0.018, (hi[1] - lo[1]) * 0.05)
    _box_from_limits(f"{prefix}0", body, lo, [hi[0], hi[1], lo[2] + deck])
    _box_from_limits(f"{prefix}1", body, [lo[0], lo[1], hi[2] - deck], hi)
    _box_from_limits(
        f"{prefix}2", body, [lo[0], lo[1], lo[2] + deck],
        [lo[0] + wall, hi[1], hi[2] - deck],
    )
    _box_from_limits(
        f"{prefix}3", body, [hi[0] - wall, lo[1], lo[2] + deck],
        [hi[0], hi[1], hi[2] - deck],
    )
    _box_from_limits(
        f"{prefix}4", body, [lo[0] + wall, hi[1] - back, lo[2] + deck],
        [hi[0] - wall, hi[1], hi[2] - deck],
    )
    for index, z_center in enumerate(shelves, start=5):
        _box_from_limits(
            f"{prefix}{index}", body,
            [lo[0] + wall, lo[1], z_center - deck * 0.5],
            [hi[0] - wall, hi[1] - back, z_center + deck * 0.5],
        )


# Assets that get a CONVEX-HULL mesh collider (a group-3 copy of the visual mesh,
# which MuJoCo collides as its convex hull) instead of a hand-built box frame:
#  - precision_machine_part: a solid machined part -> hull hugs the real outline.
#  - modular_server_drive_rack: 3 closed drive-panel faces (+X/-X/-Y) + one open face
#    (+Y); its outer envelope is box-like so the hull hugs it. (If bay INSERTION is
#    later needed, carve an open slot collider then; hull seals the open face.)
# The old box-frame colliders were built from manifest["local_bounds"], whose axis
# order is SWAPPED vs the baked visual.obj (Y<->Z) -> rotated/undersized colliders that
# did not overlap the mesh. Box-based colliders below now use the TRUE obj AABB instead.
HULL_SLUGS = ("modular_server_drive_rack", "precision_machine_part")


def _add_coacd_collision(spec, body, slug, scale, prefix, fallback_meshname):
    """Multiple convex-part mesh colliders from coacd (scripts/decompose_v2_collision.py)
    -> collision HUGS the real concave shape: flat plate top stays flat, gaps between
    machined components stay open, rack bays stay open. Each coacd part is convex so
    MuJoCo collides it exactly. Falls back to the single convex hull if not decomposed."""
    coacd_dir = os.path.join(FACTORY_V2, slug, "coacd")
    parts_json = os.path.join(coacd_dir, "parts.json")
    if not os.path.exists(parts_json):
        body.add_geom(name=f"{prefix}_hull", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=fallback_meshname, group=3, rgba=[0.5, 0.5, 0.5, 1.0],
                      condim=3, friction=[1.0, 0.01, 0.001])
        return
    count = json.load(open(parts_json))["count"]
    for n in range(count):
        obj = os.path.join(coacd_dir, f"hull_{n:02d}.obj")
        if not os.path.exists(obj):
            continue
        pm = spec.add_mesh(name=f"fa2_{slug}_ch{n}", file=obj, scale=scale)
        pm.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        body.add_geom(name=f"{prefix}_ch{n}", type=mujoco.mjtGeom.mjGEOM_MESH,
                      meshname=pm.name, group=3, rgba=[0.5, 0.5, 0.5, 1.0],
                      condim=3, friction=[1.0, 0.01, 0.001])


def _obj_aabb(obj_path, scale):
    """True min/max of the baked visual.obj vertices, scaled by the geom mesh_scale.
    This is the axis-correct extent MuJoCo actually loads (manifest local_bounds is not)."""
    lo = [float("inf")] * 3
    hi = [float("-inf")] * 3
    with open(obj_path, encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            if line.startswith("v "):
                xyz = [float(v) for v in line.split()[1:4]]
                for i in range(3):
                    lo[i] = min(lo[i], xyz[i])
                    hi[i] = max(hi[i], xyz[i])
    return ([lo[i] * scale[i] for i in range(3)], [hi[i] * scale[i] for i in range(3)])


# Per-asset uniform up-scale applied at inject time (visual + coacd colliders together),
# with a re-seat so the base stays on its support surface. History: x1.4 (2026-07-11, flat
# placement era); back to x1.0 on 2026-07-12 - the ECU must BRIDGE the U-notch mouth
# (rest on the plate top spanning the cutout), so the notch must be NARROWER than the
# 151mm ECU: at x1.0 the notch is ~125mm -> ~13mm of bearing on each side.
# RESEAT = (support_top_world_z, original_body_z).
SCALE_MULT = {"precision_machine_part": 1.0}
RESEAT = {"precision_machine_part": (0.750, 0.836)}


def inject_v2(spec):
    if not os.path.exists(MANIFEST_V2):
        raise FileNotFoundError(f"factory v2 manifest missing: {MANIFEST_V2}")
    manifest = json.load(open(MANIFEST_V2, encoding="utf-8"))
    for slug in V2_STATIC_SLUGS:
        asset = manifest[slug]
        k = SCALE_MULT.get(slug, 1.0)
        sc = [s * k for s in asset["mesh_scale"]]
        pos = list(asset["mujoco_position"])
        if slug in RESEAT:  # keep the (now larger) base resting on its support surface
            support, origz = RESEAT[slug]
            pos[2] = support + k * (origz - support)
        body = spec.worldbody.add_body(
            name=f"fa2_{slug}", pos=pos, quat=asset["mujoco_quat"]
        )
        texture = spec.add_texture(
            name=f"fa2_{slug}_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
            file=os.path.join(FACTORY_V2, asset["texture"]),
        )
        material = spec.add_material(name=f"fa2_{slug}_mat", specular=0.1, shininess=0.2)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture.name
        mesh = spec.add_mesh(
            name=f"fa2_{slug}_mesh", file=os.path.join(FACTORY_V2, asset["visual_obj"]),
            scale=sc,
        )
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        body.add_geom(
            name=f"fa2_{slug}_vis", type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname=mesh.name, material=material.name, group=1,
            contype=0, conaffinity=0, density=0,
        )
        prefix = f"fa2_{slug}_col"
        # Prefer coacd convex decomposition (scripts/decompose_v2_collision.py) for ANY
        # asset that has it -> collision hugs the true concave shape (flat tops stay flat,
        # gaps/bays stay open, no phantom full-width slabs). Fall back to the box heuristics
        # only when a slug was never decomposed.
        if os.path.exists(os.path.join(FACTORY_V2, slug, "coacd", "parts.json")):
            _add_coacd_collision(spec, body, slug, sc, prefix, mesh.name)
        else:
            lo, hi = _obj_aabb(os.path.join(FACTORY_V2, asset["visual_obj"]), sc)
            if slug == "checkout_counter":
                _add_checkout_collision(body, lo, hi, prefix)
            elif slug == "kitchen_base_cabinet":
                _add_cabinet_collision(body, lo, hi, prefix)
            else:
                _add_open_frame_collision(body, lo, hi, prefix)
    return spec


def load_model(free=False, placement=None, task_rack_slot=False, use_v2=True):
    """Scene model with factory props injected (static unless free=True)."""
    spec = mujoco.MjSpec.from_file(SCENE_XML)
    if use_v2:
        inject_v2(spec)
    else:
        inject(spec, free=free, placement=placement, task_rack_slot=task_rack_slot)
    return spec.compile()
