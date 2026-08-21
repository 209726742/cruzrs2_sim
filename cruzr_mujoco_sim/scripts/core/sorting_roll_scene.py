#!/usr/bin/env python3
"""Build and smoke-check the CRUZR table-to-shelf roll scene."""

import argparse
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
SORTING_ROOT = WORKSPACE_ROOT / "Sorting_Roll"
TEMPLATE_PATH = SORTING_ROOT / "sorting_roll_scene.xml"
SCENE_PATH = PACKAGE_ROOT / "assets" / "sorting_roll_scene.xml"

SHELF_BOUNDS = np.array([[0.8, -0.315, 0.0], [1.2, 0.315, 1.4]])
TABLE_BOUNDS = np.array([[-0.3, -1.310, 0.0], [0.3, -0.790, 1.0]])
TABLE_YAW_DEG = 180.0
ROLL_DEPTH_FRACTION_FROM_ROBOT_SIDE = 1.0 / 3.0
ROLL_SIZE = np.array([0.5, 0.025, 0.025])
ROLL_RADIUS_M = 0.012
ROLL_SUPPORT_X_M = 0.215
ROLL_SUPPORT_HALF_X_M = 0.018
ROLL_SUPPORT_TOP_Z_M = 1.100
TABLE_DEPTH_M = float(TABLE_BOUNDS[1, 1] - TABLE_BOUNDS[0, 1])
ROLL_SPAWN = np.array([
    0.0,
    TABLE_BOUNDS[1, 1]
    - TABLE_DEPTH_M * ROLL_DEPTH_FRACTION_FROM_ROBOT_SIDE,
    ROLL_SUPPORT_TOP_Z_M + ROLL_RADIUS_M + 0.0015,
])
TARGET_CENTER = np.array([0.7825, 0.0, 1.0125])
TARGET_AXIS = np.array([0.0, 1.0, 0.0])
EXPECTED_EDGE_GAP_M = 0.475


def required_assets():
    return (
        TEMPLATE_PATH,
        PACKAGE_ROOT / "assets" / "cruzr_pgc140.xml",
        SORTING_ROOT / "Assets" / "Roll" / "Meshy_AI__0819075833_texture_obj"
        / "Meshy_AI__0819075833_texture_lowpoly.obj",
        SORTING_ROOT / "Assets" / "Roll" / "Meshy_AI__0819075833_texture_obj"
        / "Meshy_AI__0819075833_texture.png",
        SORTING_ROOT / "Assets" / "Shelf"
        / "Meshy_AI_Corrected_Tiered_Shel_0819090318_image-to-3d-texture_obj"
        / "Meshy_AI_Corrected_Tiered_Shel_0819090318_image-to-3d-texture_lowpoly.obj",
        SORTING_ROOT / "Assets" / "Shelf"
        / "Meshy_AI_Corrected_Tiered_Shel_0819090318_image-to-3d-texture_obj"
        / "Meshy_AI_Corrected_Tiered_Shel_0819090318_image-to-3d-texture.png",
        SORTING_ROOT / "Assets" / "Table" / "Meshy_AI__0819090550_texture_obj"
        / "Meshy_AI__0819090550_texture_lowpoly.obj",
        SORTING_ROOT / "Assets" / "Table" / "Meshy_AI__0819090550_texture_obj"
        / "Meshy_AI__0819090550_texture.png",
    )


def layout_report():
    edge_gap = SHELF_BOUNDS[0, 1] - TABLE_BOUNDS[1, 1]
    roll_depth_from_robot_side = TABLE_BOUNDS[1, 1] - ROLL_SPAWN[1]
    shelf_size = SHELF_BOUNDS[1] - SHELF_BOUNDS[0]
    table_size = TABLE_BOUNDS[1] - TABLE_BOUNDS[0]
    checks = {
        "shelf_size_0p63_x_0p40_x_1p40": bool(
            np.allclose(shelf_size, [0.4, 0.63, 1.4], atol=1e-9)
        ),
        "table_size_0p60_x_0p52_x_1p00": bool(
            np.allclose(table_size, [0.6, 0.52, 1.0], atol=1e-9)
        ),
        "table_is_robot_right": bool(TABLE_BOUNDS[1, 1] < 0.0),
        "shelf_is_robot_front": bool(SHELF_BOUNDS[0, 0] > 0.0),
        "edge_gap_is_0p475m": bool(abs(edge_gap - EXPECTED_EDGE_GAP_M) < 1e-9),
        "table_yaw_is_180deg": bool(abs(TABLE_YAW_DEG - 180.0) < 1e-9),
        "roll_at_robot_side_third": bool(
            abs(
                roll_depth_from_robot_side
                - TABLE_DEPTH_M * ROLL_DEPTH_FRACTION_FROM_ROBOT_SIDE
            ) < 1e-9
        ),
        "roll_starts_above_pickup_support": bool(
            TABLE_BOUNDS[0, 0] <= ROLL_SPAWN[0] <= TABLE_BOUNDS[1, 0]
            and TABLE_BOUNDS[0, 1] <= ROLL_SPAWN[1] <= TABLE_BOUNDS[1, 1]
            and ROLL_SPAWN[2] > ROLL_SUPPORT_TOP_Z_M
        ),
        "pickup_supports_leave_center_grasp_clear": bool(
            ROLL_SUPPORT_X_M - ROLL_SUPPORT_HALF_X_M >= 0.19
            and ROLL_SUPPORT_X_M + ROLL_SUPPORT_HALF_X_M <= ROLL_SIZE[0] / 2.0
        ),
        "target_is_top_tier": bool(1.0 <= TARGET_CENTER[2] <= 1.2),
        "target_axis_is_shelf_width": bool(np.allclose(TARGET_AXIS, [0.0, 1.0, 0.0])),
    }
    return {
        "shelf_bounds_m": SHELF_BOUNDS.tolist(),
        "table_bounds_m": TABLE_BOUNDS.tolist(),
        "roll_size_m": ROLL_SIZE.tolist(),
        "roll_spawn_m": ROLL_SPAWN.tolist(),
        "roll_support_x_m": [-ROLL_SUPPORT_X_M, ROLL_SUPPORT_X_M],
        "roll_support_top_z_m": ROLL_SUPPORT_TOP_Z_M,
        "table_yaw_deg": TABLE_YAW_DEG,
        "roll_depth_from_robot_side_m": float(roll_depth_from_robot_side),
        "roll_depth_fraction_from_robot_side": (
            float(roll_depth_from_robot_side / TABLE_DEPTH_M)
        ),
        "target_center_m": TARGET_CENTER.tolist(),
        "edge_gap_m": float(edge_gap),
        "checks": checks,
    }


def materialize_scene(destination=SCENE_PATH):
    missing = [str(path) for path in required_assets() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Sorting Roll assets:\n" + "\n".join(missing))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(TEMPLATE_PATH, destination)
    return destination


def _object_id(mujoco, model, object_type, name):
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise RuntimeError(f"scene is missing {name}")
    return object_id


def smoke_check(scene_path, steps=1000):
    import mujoco

    started = time.monotonic()
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    load_seconds = time.monotonic() - started
    data = mujoco.MjData(model)

    roll_body = _object_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "sorting_roll"
    )
    roll_joint = _object_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, "sorting_roll_free"
    )
    roll_geom = _object_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, "sorting_roll_col"
    )
    support_geoms = {
        _object_id(mujoco, model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in (
            "roll_support_x_negative_base_col",
            "roll_support_x_positive_base_col",
            "roll_support_x_negative_robot_lip_col",
            "roll_support_x_negative_far_lip_col",
            "roll_support_x_positive_robot_lip_col",
            "roll_support_x_positive_far_lip_col",
        )
    }
    target_site = _object_id(
        mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "sorting_roll_target"
    )

    mujoco.mj_forward(model, data)
    for _ in range(steps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    supported = False
    for contact_index in range(data.ncon):
        pair = {int(data.contact[contact_index].geom1), int(data.contact[contact_index].geom2)}
        if roll_geom in pair and pair & support_geoms:
            supported = True
            break

    dof_adr = int(model.jnt_dofadr[roll_joint])
    roll_speed = float(np.linalg.norm(data.qvel[dof_adr:dof_adr + 6]))
    roll_position = data.xpos[roll_body].copy()
    target_position = data.site_xpos[target_site].copy()
    checks = {
        "finite_state": bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
        ),
        "roll_supported_by_pickup_stand": supported,
        "roll_remains_on_pickup_stand": bool(
            TABLE_BOUNDS[0, 0] <= roll_position[0] <= TABLE_BOUNDS[1, 0]
            and abs(float(roll_position[1] - ROLL_SPAWN[1])) <= 0.025
            and ROLL_SUPPORT_TOP_Z_M + ROLL_RADIUS_M - 0.005
            <= roll_position[2]
            <= ROLL_SUPPORT_TOP_Z_M + ROLL_RADIUS_M + 0.02
        ),
        "roll_settled": bool(roll_speed < 0.05),
        "target_site_matches_contract": bool(
            np.allclose(target_position, TARGET_CENTER, atol=1e-6)
        ),
        "robot_actuator_contract_preserved": bool(model.nu == 19),
    }
    return model, data, {
        "scene_path": str(Path(scene_path).resolve()),
        "load_seconds": round(load_seconds, 3),
        "simulated_seconds": round(float(data.time), 3),
        "model": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "nmesh": int(model.nmesh),
            "nmeshvert": int(model.nmeshvert),
            "nmeshface": int(model.nmeshface),
        },
        "roll_position_m": np.round(roll_position, 6).tolist(),
        "roll_speed": round(roll_speed, 6),
        "target_position_m": np.round(target_position, 6).tolist(),
        "checks": checks,
    }


def render_preview(
    model,
    data,
    output_path,
    *,
    lookat=(0.45, -0.72, 0.72),
    distance=3.35,
    azimuth=42,
    elevation=-24,
):
    import imageio.v3 as iio
    import mujoco

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    renderer = mujoco.Renderer(model, height=720, width=1280)
    try:
        renderer.update_scene(data, camera=camera)
        iio.imwrite(output_path, renderer.render())
    finally:
        renderer.close()
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--target-steps", type=int, default=1200)
    parser.add_argument("--render", help="optional preview PNG path")
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.target_steps < 1:
        parser.error("--target-steps must be positive")

    layout = layout_report()
    if not all(layout["checks"].values()):
        raise SystemExit(json.dumps(layout, indent=2))
    scene_path = materialize_scene()
    if args.build_only:
        print(scene_path)
        return 0

    if args.render:
        os.environ.setdefault("MUJOCO_GL", "egl")
    model, data, runtime = smoke_check(scene_path, steps=args.steps)
    from sorting_roll_task import (
        evaluate_placement,
        fit_report,
        target_placement_smoke,
    )

    initial_task = evaluate_placement(model, data)
    target_data, target = target_placement_smoke(model, steps=args.target_steps)
    report = {
        "layout": layout,
        "fit": fit_report(),
        "runtime": runtime,
        "initial_task_state": initial_task,
        "target_placement": target,
    }
    if args.render:
        initial_preview = render_preview(model, data, args.render)
        target_preview = initial_preview.with_name(
            initial_preview.stem + "_target" + initial_preview.suffix
        )
        render_preview(
            model,
            target_data,
            target_preview,
            lookat=(0.90, 0.0, 1.03),
            distance=1.15,
            azimuth=42,
            elevation=-12,
        )
        report["preview"] = {
            "initial": str(initial_preview.resolve()),
            "target": str(target_preview.resolve()),
        }
    print(json.dumps(report, indent=2))
    passed = (
        all(runtime["checks"].values())
        and not initial_task["instantaneous_success"]
        and target["success"]
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
