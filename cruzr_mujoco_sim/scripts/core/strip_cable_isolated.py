#!/usr/bin/env python3
"""Build and validate an isolated cable using fitted strip parameters.

This stage deliberately uses zero gravity and no contacts.  It verifies that
the pre-curved OBJ centreline is a zero-stress reference before damping is fit
or the object is introduced into the robot scene.
"""

import argparse
import hashlib
import json
import os

import mujoco
import numpy as np

from shelf_e2e_objects import object_info
from strip_cable_structure import DEFAULT_OBJ, NODE_COUNT, load_strip_geometry, sample_nodes


SCHEMA_VERSION = 1
DEFAULT_DURATION_S = 10.0
DEFAULT_TIMESTEP_S = 0.001
GEOMETRY_TOLERANCE_MM = 1.2
REFERENCE_FORCE_TOLERANCE_N = 1e-8
REFERENCE_DRIFT_TOLERANCE = 1e-9


def load_parameter_candidate(path):
    with open(path, "rb") as handle:
        raw = handle.read()
    document = json.loads(raw.decode("utf-8"))
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported strip parameter schema")
    if document.get("mode") != "isolated_cable_parameter_candidate":
        raise ValueError("input is not an isolated cable parameter candidate")
    if document.get("fit_ready_for_isolated_dynamics") is not True:
        raise ValueError("parameter candidate did not pass material-fit gates")
    required = {
        "mass_kg": document["geometry"]["mass_kg"],
        "width_m": document["geometry"]["width_m"],
        "thickness_m": document["geometry"]["thickness_m"],
        "bend_pa": document["mujoco"]["bend_config_pa"],
        "twist_pa": document["mujoco"]["twist_config_pa"],
    }
    if not all(np.isfinite(value) and float(value) > 0.0 for value in required.values()):
        raise ValueError("candidate contains a non-positive physical parameter")
    if document["mujoco"].get("flat") is not False:
        raise ValueError("pre-curved strip requires flat=false")
    return document, hashlib.sha256(raw).hexdigest()


def build_isolated_xml(parameters, *, obj_path=DEFAULT_OBJ, damping=0.0,
                       free_root=True):
    damping = float(damping)
    if not np.isfinite(damping) or damping < 0.0:
        raise ValueError("joint damping must be finite and non-negative")

    geometry = load_strip_geometry(obj_path)
    nodes = sample_nodes(geometry)
    segment_length = float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).mean())
    vertex = " ".join(" ".join(f"{value:.12g}" for value in row) for row in nodes)
    mass = float(parameters["geometry"]["mass_kg"])
    width = float(parameters["geometry"]["width_m"])
    thickness = float(parameters["geometry"]["thickness_m"])
    bend = float(parameters["mujoco"]["bend_config_pa"])
    twist = float(parameters["mujoco"]["twist_config_pa"])
    friction = float(parameters["contact"]["shelf_mu_mean"])
    root_joint = '<freejoint name="strip_free"/>' if free_root else ""

    xml = f"""
<mujoco model="strip_cable_isolated">
  <option timestep="{DEFAULT_TIMESTEP_S:.9g}" integrator="implicitfast" gravity="0 0 0"/>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <worldbody>
    <body name="strip">
      {root_joint}
      <composite prefix="strip_" type="cable" initial="none" vertex="{vertex}">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="{twist:.12g}"/>
          <config key="bend" value="{bend:.12g}"/>
          <config key="flat" value="false"/>
        </plugin>
        <joint kind="main" damping="{damping:.12g}"/>
        <geom type="box"
              size="{0.5 * segment_length:.12g} {0.5 * thickness:.12g} {0.5 * width:.12g}"
              mass="{mass / (NODE_COUNT - 1):.12g}" group="3" condim="4"
              solref="0.004 1" solimp="0.95 0.995 0.001"
              friction="{friction:.12g} 0.02 0.001"/>
        <skin rgba="0.1 0.1 0.11 1" subgrid="3"/>
      </composite>
    </body>
  </worldbody>
</mujoco>
"""
    return xml, geometry, nodes


def validate_isolated(parameters, *, obj_path=DEFAULT_OBJ, damping=0.0,
                      duration_s=DEFAULT_DURATION_S):
    duration_s = float(duration_s)
    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration must be finite and positive")
    xml, geometry, nodes = build_isolated_xml(
        parameters, obj_path=obj_path, damping=damping
    )
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    info = object_info(model, "strip")

    lo, hi = [], []
    for geom in sorted(info["geoms"]):
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise RuntimeError("isolated cable generated a non-box collision geom")
        half = np.abs(data.geom_xmat[geom].reshape(3, 3)) @ model.geom_size[geom]
        lo.append(data.geom_xpos[geom] - half)
        hi.append(data.geom_xpos[geom] + half)
    cable_lo = np.min(lo, axis=0)
    cable_hi = np.max(hi, axis=0)
    bbox_error_mm = 1000.0 * np.maximum(
        np.abs(cable_lo - geometry["bbox_lo_m"]),
        np.abs(cable_hi - geometry["bbox_hi_m"]),
    )

    initial_qpos = data.qpos.copy()
    initial_passive_force = float(np.max(np.abs(data.qfrc_passive), initial=0.0))
    max_qvel = 0.0
    finite = True
    steps = int(round(duration_s / model.opt.timestep))
    for _ in range(steps):
        mujoco.mj_step(model, data)
        finite = finite and bool(
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
        )
        max_qvel = max(max_qvel, float(np.max(np.abs(data.qvel), initial=0.0)))
        if not finite:
            break
    qpos_drift = float(np.max(np.abs(data.qpos - initial_qpos), initial=0.0))
    steps_completed = int(round(data.time / model.opt.timestep))

    target_mass = float(parameters["geometry"]["mass_kg"])
    checks = {
        "segments_14": len(info["geoms"]) == 14,
        "internal_ball_joints_13": len(info["ball_qpos_adrs"]) == 13,
        "adjacent_exclusions_13": model.nexclude == 13,
        "single_skin": model.nskin == 1,
        "mass_matches_candidate": abs(info["mass_kg"] - target_mass) < 1e-9,
        "geometry_within_1p2mm": bool(np.max(bbox_error_mm) <= GEOMETRY_TOLERANCE_MM),
        "reference_passive_force_near_zero": initial_passive_force <= REFERENCE_FORCE_TOLERANCE_N,
        "finite_for_requested_duration": finite and steps_completed == steps,
        "reference_shape_no_drift": qpos_drift <= REFERENCE_DRIFT_TOLERANCE,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "mode": "isolated_cable_static_validation",
        "physical_parameters": True,
        "source_measurement_provenance": parameters["source_measurement_provenance"],
        "formal_collection_allowed": bool(parameters["formal_collection_allowed"]),
        "static_ready_for_damping_fit": all(checks.values()),
        "configuration": {
            "obj": os.path.abspath(obj_path),
            "duration_s": duration_s,
            "timestep_s": float(model.opt.timestep),
            "damping_nms_per_rad": float(damping),
            "bend_pa": float(parameters["mujoco"]["bend_config_pa"]),
            "twist_pa": float(parameters["mujoco"]["twist_config_pa"]),
        },
        "compiled": {
            "nq": int(model.nq),
            "nv": int(model.nv),
            "segments": len(info["geoms"]),
            "internal_ball_joints": len(info["ball_qpos_adrs"]),
            "skins": int(model.nskin),
            "adjacent_exclusions": int(model.nexclude),
            "mass_kg": info["mass_kg"],
            "bbox_abs_error_mm": bbox_error_mm.tolist(),
            "nodes": nodes.tolist(),
        },
        "stability": {
            "steps_requested": steps,
            "steps_completed": steps_completed,
            "initial_max_abs_passive_force_n": initial_passive_force,
            "max_abs_qvel": max_qvel,
            "max_abs_qpos_drift": qpos_drift,
            "finite": finite,
        },
        "checks": checks,
    }
    return xml, model, report


def _write_new(path, payload):
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite existing output: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_json")
    parser.add_argument("--obj", default=DEFAULT_OBJ)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--xml-output")
    parser.add_argument("--report-output")
    args = parser.parse_args(argv)

    parameters, parameter_sha256 = load_parameter_candidate(args.parameter_json)
    xml, _, report = validate_isolated(
        parameters, obj_path=args.obj, duration_s=args.duration_s
    )
    report["source_parameter_sha256"] = parameter_sha256
    report_payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.xml_output:
        _write_new(args.xml_output, xml)
    if args.report_output:
        _write_new(args.report_output, report_payload)
    print(report_payload, end="")
    raise SystemExit(0 if report["static_ready_for_damping_fit"] else 1)


if __name__ == "__main__":
    main()
