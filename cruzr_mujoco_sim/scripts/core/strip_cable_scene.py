#!/usr/bin/env python3
"""Generate and validate an isolated flexible variant of the dual shelf scene."""

import argparse
import hashlib
import json
import math
import os
import re
import tempfile

import mujoco
import numpy as np

from shelf_e2e_flex_state import (
    FLEX_TASK_VERSION,
    RIGID_TASK_VERSION,
    object_state_contract,
)
from shelf_e2e_objects import internal_ball_quaternions, object_info
from strip_cable_structure import NODE_COUNT, load_strip_geometry, sample_nodes


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS_DIR)
DEFAULT_TEMPLATE = os.path.join(ROOT, "assets", "e2e", "template_pillar_v1.xml")
DEFAULT_OBJ = os.path.join(ROOT, "assets", "shelf", "meshes", "RubberStrip.obj")
RIGID_MODEL_ID = "rigid_segmented_v1"
SYNTHETIC_FLEX_MODEL_ID = "strip_cable_synthetic_v1"
SYNTHETIC_FLEX_V2_MODEL_ID = "strip_cable_synthetic_v2_reinforced"
PAD_NAMES = ("L_pad1", "L_pad2", "R_pad1", "R_pad2")
# cruzr_pgc140.xml declares compiler boundmass=0.0001.  The cable composite
# leaves its free root empty, so the compiler assigns that root exactly 0.1 g.
SCENE_EMPTY_ROOT_BOUND_MASS_KG = 0.0001
STRIP_BODY_PATTERN = re.compile(
    r'(?P<indent>    )<body name="strip" pos="(?P<pos>[^"]+)">.*?^    </body>\n',
    re.MULTILINE | re.DOTALL,
)


def load_calibrated_parameters(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    document = json.loads(raw.decode("utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("unsupported calibrated parameter schema")
    if document.get("mode") != "isolated_cable_parameter_calibrated":
        raise ValueError("scene generation requires calibrated cable parameters")
    if document.get("fit_ready_for_scene_template") is not True:
        raise ValueError("calibrated parameters did not pass the damping-fit gates")
    damping = document.get("mujoco", {}).get("joint_damping_nms_per_rad")
    if not isinstance(damping, (int, float)) or not math.isfinite(damping) or damping <= 0.0:
        raise ValueError("calibrated parameters need positive joint damping")
    return document, hashlib.sha256(raw).hexdigest()


def select_scene_template(model_id, *, norec):
    """Return a verified template/manifest; synthetic flex cannot record."""
    if model_id == RIGID_MODEL_ID:
        return DEFAULT_TEMPLATE, {
            "schema_version": 1,
            "model_id": RIGID_MODEL_ID,
            "task_version": RIGID_TASK_VERSION,
            "formal_collection_allowed": True,
        }
    template_names = {
        SYNTHETIC_FLEX_MODEL_ID: "template_strip_cable_v1.xml",
        SYNTHETIC_FLEX_V2_MODEL_ID: "template_strip_cable_v2_reinforced.xml",
    }
    if model_id not in template_names:
        raise ValueError(f"unsupported E2E_STRIP_MODEL: {model_id!r}")
    template = os.path.join(ROOT, "assets", "e2e", template_names[model_id])
    manifest_path = os.path.join(
        ROOT, "assets", "e2e", os.path.splitext(template_names[model_id])[0] + ".manifest.json"
    )
    with open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    with open(template, "rb") as handle:
        template_sha = hashlib.sha256(handle.read()).hexdigest()
    if manifest.get("model_id") != model_id:
        raise ValueError("flex scene manifest model_id mismatch")
    if manifest.get("task_version") != FLEX_TASK_VERSION:
        raise ValueError("flex scene manifest task_version mismatch")
    if manifest.get("scene_ready_for_norec") is not True:
        raise ValueError("flex scene manifest is not ready for NOREC")
    if manifest.get("template_sha256") != template_sha:
        raise ValueError("flex scene template SHA-256 does not match its manifest")
    if not norec and manifest.get("formal_collection_allowed") is not True:
        raise ValueError(
            "synthetic flexible strip is NOREC-only; physical measurements are "
            "required before recording"
        )
    return template, manifest


def _strip_body(parameters, obj_path, position):
    geometry = load_strip_geometry(obj_path)
    nodes = sample_nodes(geometry)
    segment_length = float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).mean())
    vertex = " ".join(" ".join(f"{value:.12g}" for value in row) for row in nodes)
    physical = parameters["geometry"]
    mujoco_config = parameters["mujoco"]
    shelf_mu = float(parameters["contact"]["shelf_mu_mean"])
    return f'''    <body name="strip" pos="{position}">
      <freejoint name="strip_free"/>
      <composite prefix="strip_" type="cable" initial="none" vertex="{vertex}">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="{float(mujoco_config['twist_config_pa']):.12g}"/>
          <config key="bend" value="{float(mujoco_config['bend_config_pa']):.12g}"/>
          <config key="flat" value="false"/>
        </plugin>
        <joint kind="main" damping="{float(mujoco_config['joint_damping_nms_per_rad']):.12g}"/>
        <geom type="box"
              size="{0.5 * segment_length:.12g} {0.5 * float(physical['thickness_m']):.12g} {0.5 * float(physical['width_m']):.12g}"
              mass="{(float(physical['mass_kg']) - SCENE_EMPTY_ROOT_BOUND_MASS_KG) / (NODE_COUNT - 1):.12g}"
              group="3" rgba="0.9 0.5 0.1 0.5" condim="4" priority="1"
              solref="0.004 4" solimp="0.95 0.995 0.001"
              friction="{shelf_mu:.12g} 0.02 0.001"/>
        <skin material="strip_mat" subgrid="3"/>
      </composite>
    </body>
'''


def _pad_contact_pairs(parameters):
    pad_mu = float(parameters["contact"]["gripper_pad_mu_mean"])
    lines = ["  <contact>"]
    for segment in range(NODE_COUNT - 1):
        for pad in PAD_NAMES:
            lines.append(
                f'    <pair geom1="strip_G{segment}" geom2="{pad}" condim="6" '
                f'solref="0.0015 1" solimp="0.98 0.9995 0.0005" '
                f'friction="{pad_mu:.12g} {pad_mu:.12g} 0.02 0.001 0.001"/>'
            )
    lines.append("  </contact>")
    return "\n".join(lines) + "\n"


def build_scene(template_path, parameters, obj_path=DEFAULT_OBJ):
    with open(template_path, encoding="utf-8") as handle:
        template = handle.read()
    matches = list(STRIP_BODY_PATTERN.finditer(template))
    if len(matches) != 1:
        raise ValueError(f"base template must contain one rigid strip body, got {len(matches)}")
    if "mujoco.elasticity.cable" in template:
        raise ValueError("base template already contains the cable plugin")
    include = '<include file="cruzr_pgc140.xml"/>'
    if template.count(include) != 1:
        raise ValueError("base template include anchor is missing or duplicated")
    scene = template.replace(
        include,
        include + '\n  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>',
        1,
    )
    scene, replacements = STRIP_BODY_PATTERN.subn(
        lambda match: _strip_body(parameters, obj_path, match.group("pos")),
        scene,
        count=1,
    )
    if replacements != 1:
        raise RuntimeError("rigid strip replacement failed")
    closing = "  </worldbody>\n</mujoco>"
    if scene.count(closing) != 1:
        raise ValueError("base template closing anchor is missing or duplicated")
    scene = scene.replace(
        closing,
        "  </worldbody>\n" + _pad_contact_pairs(parameters) + "</mujoco>",
        1,
    )
    return scene


def _compile_text(text, assets_root):
    path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", prefix=".strip_scene_",
            dir=assets_root, encoding="utf-8", delete=False,
        ) as handle:
            handle.write(text)
            path = handle.name
        return mujoco.MjModel.from_xml_path(path)
    finally:
        if path and os.path.exists(path):
            os.unlink(path)


def _names(model, object_type, count):
    return tuple(
        mujoco.mj_id2name(model, object_type, index)
        for index in range(count)
    )


def _strip_dofs(model, strip_info):
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "strip_free")]
    joints.extend(strip_info["ball_joints"])
    dofs = []
    for joint in joints:
        start = int(model.jnt_dofadr[joint])
        width = 6 if int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_FREE) else 3
        dofs.extend(range(start, start + width))
    return np.asarray(dofs, dtype=np.int32)


def validate_scene(scene, template_path, parameters, *, assets_root, settle_s=2.0):
    with open(template_path, encoding="utf-8") as handle:
        rigid_text = handle.read()
    rigid = _compile_text(rigid_text, assets_root)
    model = _compile_text(scene, assets_root)
    objects = {name: object_info(model, name) for name in ("pillar", "strip")}
    task_version, contract = object_state_contract(objects)
    strip = objects["strip"]

    flex_joint_names = set(strip["ball_joint_names"])
    rigid_joints = _names(rigid, mujoco.mjtObj.mjOBJ_JOINT, rigid.njnt)
    flex_noninternal_joints = tuple(
        name for name in _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
        if name not in flex_joint_names
    )
    interface_equal = {
        "noninternal_joints": rigid_joints == flex_noninternal_joints,
        "actuators": _names(rigid, mujoco.mjtObj.mjOBJ_ACTUATOR, rigid.nu)
        == _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
        "sensors": _names(rigid, mujoco.mjtObj.mjOBJ_SENSOR, rigid.nsensor)
        == _names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
        "cameras": _names(rigid, mujoco.mjtObj.mjOBJ_CAMERA, rigid.ncam)
        == _names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam),
    }

    damping_target = float(parameters["mujoco"]["joint_damping_nms_per_rad"])
    damping_values = []
    for joint in strip["ball_joints"]:
        start = int(model.jnt_dofadr[joint])
        damping_values.extend(model.dof_damping[start:start + 3])
    strip_geoms = set(strip["geoms"])
    strip_geom_names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        for geom in strip_geoms
    }
    pad_pair_friction = []
    pad_pair_count = 0
    for pair in range(model.npair):
        first = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom1[pair])
        )
        second = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, int(model.pair_geom2[pair])
        )
        if (first in strip_geom_names and second in PAD_NAMES) or (
            second in strip_geom_names and first in PAD_NAMES
        ):
            pad_pair_count += 1
            pad_pair_friction.append(model.pair_friction[pair, :2].copy())

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    strip_dofs = _strip_dofs(model, strip)
    steps = int(round(float(settle_s) / model.opt.timestep))
    finite = True
    max_final_speed = math.inf
    final_window = max(1, int(round(0.25 / model.opt.timestep)))
    final_speeds = []
    strip_bodies = sorted({int(model.geom_bodyid[geom]) for geom in strip_geoms})
    final_linear_speeds = []
    final_angular_speeds = []
    final_segment_positions = []
    for step in range(steps):
        mujoco.mj_step(model, data)
        finite = finite and bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
        )
        if step >= steps - final_window:
            final_speeds.append(float(np.max(np.abs(data.qvel[strip_dofs]))))
            final_linear_speeds.append(max(
                float(np.linalg.norm(data.cvel[body, 3:]))
                for body in strip_bodies
            ))
            final_angular_speeds.append(max(
                float(np.linalg.norm(data.cvel[body, :3]))
                for body in strip_bodies
            ))
            final_segment_positions.append(np.array([
                data.geom_xpos[geom].copy() for geom in strip_geoms
            ]))
        if not finite:
            break
    if final_speeds:
        max_final_speed = max(final_speeds)
    max_final_linear_speed = (
        max(final_linear_speeds) if final_linear_speeds else math.inf
    )
    max_final_angular_speed = (
        max(final_angular_speeds) if final_angular_speeds else math.inf
    )
    if final_segment_positions:
        position_window = np.asarray(final_segment_positions)
        segment_position_span = float(np.max(np.linalg.norm(
            np.ptp(position_window, axis=0), axis=1
        )))
    else:
        segment_position_span = math.inf
    mujoco.mj_forward(model, data)
    strip_contacts = 0
    for index in range(data.ncon):
        contact = data.contact[index]
        if int(contact.geom1) in strip_geoms or int(contact.geom2) in strip_geoms:
            strip_contacts += 1
    quaternion_norms = np.linalg.norm(
        internal_ball_quaternions(data, strip).reshape(-1, 4), axis=1
    )
    section_centres = sorted(
        (float(data.geom_xpos[geom, 1]), float(data.geom_xpos[geom, 2]))
        for geom in strip_geoms
    )
    endpoint_z = 0.5 * (section_centres[0][1] + section_centres[-1][1])
    middle_z = 0.5 * (section_centres[6][1] + section_centres[7][1])
    settled_arch_height = middle_z - endpoint_z
    warning_count = int(sum(int(item.number) for item in data.warning))

    shelf_mu = float(parameters["contact"]["shelf_mu_mean"])
    pad_mu = float(parameters["contact"]["gripper_pad_mu_mean"])
    strip_friction = model.geom_friction[list(strip_geoms), 0]
    strip_priorities = model.geom_priority[list(strip_geoms)]
    checks = {
        "flex_task_contract": task_version == FLEX_TASK_VERSION and contract is not None,
        "segments_14": len(strip_geoms) == 14,
        "internal_ball_joints_13": len(strip["ball_joints"]) == 13,
        "mass_matches_calibrated": abs(
            strip["mass_kg"] - float(parameters["geometry"]["mass_kg"])
        ) < 1e-9,
        "joint_damping_matches": bool(np.allclose(damping_values, damping_target)),
        "shelf_friction_priority_matches": bool(
            np.allclose(strip_friction, shelf_mu) and np.all(strip_priorities == 1)
        ),
        "pad_pairs_56": pad_pair_count == 56,
        "pad_pair_friction_matches": bool(
            len(pad_pair_friction) == 56 and np.allclose(pad_pair_friction, pad_mu)
        ),
        "robot_and_sensor_interfaces_unchanged": all(interface_equal.values()),
        "settle_finite": finite,
        "settle_no_mujoco_warning": warning_count == 0,
        "settle_strip_has_support_contact": strip_contacts > 0,
        "settle_internal_quaternions_unit": bool(
            np.allclose(quaternion_norms, 1.0, atol=1e-6, rtol=0.0)
        ),
        "settle_segment_linear_speed_below_0p02": max_final_linear_speed <= 0.02,
        "settle_segment_angular_speed_below_0p1": max_final_angular_speed <= 0.1,
        "settle_segment_position_span_below_0p1mm": segment_position_span <= 0.0001,
        "settle_pickup_arch_height_above_30mm": settled_arch_height >= 0.030,
    }
    return {
        "schema_version": 1,
        "mode": "dual_scene_flexible_strip_validation",
        "task_version": task_version,
        "formal_collection_allowed": bool(parameters["formal_collection_allowed"]),
        "scene_ready_for_norec": all(checks.values()),
        "compiled": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "nu": int(model.nu),
            "ncam": int(model.ncam),
            "nsensor": int(model.nsensor),
            "strip_segments": len(strip_geoms),
            "strip_ball_joints": len(strip["ball_joints"]),
            "strip_mass_kg": strip["mass_kg"],
            "pad_pair_count": pad_pair_count,
            "interface_equal": interface_equal,
        },
        "settle": {
            "duration_s": float(settle_s),
            "steps": steps,
            "finite": finite,
            "mujoco_warning_count": warning_count,
            "strip_contact_count_final": strip_contacts,
            "max_abs_strip_qvel_last_0p25s": max_final_speed,
            "max_segment_linear_speed_mps_last_0p25s": max_final_linear_speed,
            "max_segment_angular_speed_radps_last_0p25s": max_final_angular_speed,
            "max_segment_position_span_m_last_0p25s": segment_position_span,
            "pickup_arch_height_m": settled_arch_height,
            "internal_quaternion_norm_range": [
                float(np.min(quaternion_norms)), float(np.max(quaternion_norms))
            ],
        },
        "checks": checks,
    }


def _write_new(path, payload):
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite existing output: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_json")
    parser.add_argument("--template", default=DEFAULT_TEMPLATE)
    parser.add_argument("--obj", default=DEFAULT_OBJ)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--model-id", default=SYNTHETIC_FLEX_MODEL_ID)
    parser.add_argument("--settle-s", type=float, default=2.0)
    args = parser.parse_args(argv)

    parameters, parameter_sha = load_calibrated_parameters(args.parameter_json)
    scene = build_scene(args.template, parameters, args.obj)
    report = validate_scene(
        scene,
        args.template,
        parameters,
        assets_root=os.path.join(ROOT, "assets"),
        settle_s=args.settle_s,
    )
    with open(args.template, "rb") as handle:
        template_sha = hashlib.sha256(handle.read()).hexdigest()
    scene_sha = hashlib.sha256(scene.encode("utf-8")).hexdigest()
    manifest = {
        "schema_version": 1,
        "model_id": args.model_id,
        "task_version": FLEX_TASK_VERSION,
        "source_template_sha256": template_sha,
        "source_parameter_sha256": parameter_sha,
        "template_sha256": scene_sha,
        "source_measurement_provenance": parameters["source_measurement_provenance"],
        "formal_collection_allowed": bool(parameters["formal_collection_allowed"]),
        "scene_ready_for_norec": report["scene_ready_for_norec"],
    }
    report.update({
        "source_template_sha256": template_sha,
        "source_parameter_sha256": parameter_sha,
        "template_sha256": scene_sha,
    })
    if report["scene_ready_for_norec"]:
        _write_new(args.output, scene)
        _write_new(args.manifest_output, json.dumps(manifest, indent=2) + "\n")
    _write_new(args.report_output, json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    raise SystemExit(0 if report["scene_ready_for_norec"] else 1)


if __name__ == "__main__":
    main()
