#!/usr/bin/env python3
"""Compile-only structure probe for the pre-curved rubber-strip cable model.

This tool intentionally does not write XML or advance dynamics. Its bend, twist,
and damping values are non-physical sentinels used only to verify model topology.
"""

import argparse
import json
import os

import mujoco
import numpy as np

from shelf_e2e_objects import object_info


HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS_DIR)
DEFAULT_OBJ = os.path.join(ROOT, "assets", "shelf", "meshes", "RubberStrip.obj")
NODE_COUNT = 15
REFERENCE_MASS_KG = 0.4004
SENTINEL_BEND = 1.0
SENTINEL_TWIST = 1.0
SENTINEL_DAMPING = 0.0
GEOMETRY_TOLERANCE_MM = 1.1


def load_strip_geometry(path):
    vertices = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("v "):
                vertices.append([float(value) for value in line.split()[1:4]])
    vertices = np.asarray(vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError(f"invalid OBJ vertices in {path}")

    section_y, counts = np.unique(vertices[:, 1], return_counts=True)
    if len(section_y) < NODE_COUNT or not np.all(counts == 4):
        raise ValueError("RubberStrip OBJ must contain four vertices per longitudinal section")
    centres = np.asarray([
        np.mean(vertices[vertices[:, 1] == y], axis=0)
        for y in section_y
    ])
    widths = np.asarray([
        np.ptp(vertices[vertices[:, 1] == y, 0])
        for y in section_y
    ])
    thicknesses = np.asarray([
        np.ptp(vertices[vertices[:, 1] == y, 2])
        for y in section_y
    ])
    return {
        "vertices": vertices,
        "section_y": section_y,
        "centres": centres,
        "width_m": float(np.median(widths)),
        "thickness_m": float(np.median(thicknesses)),
        "bbox_lo_m": vertices.min(axis=0),
        "bbox_hi_m": vertices.max(axis=0),
        "centerline_arc_m": float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum()),
    }


def sample_nodes(geometry, count=NODE_COUNT):
    y = np.linspace(geometry["section_y"][0], geometry["section_y"][-1], count)
    centres = geometry["centres"]
    x = np.interp(y, centres[:, 1], centres[:, 0])
    z = np.interp(y, centres[:, 1], centres[:, 2])
    return np.column_stack([x, y, z])


def build_structure_probe_xml(obj_path=DEFAULT_OBJ):
    geometry = load_strip_geometry(obj_path)
    nodes = sample_nodes(geometry)
    segment_length = float(np.linalg.norm(np.diff(nodes, axis=0), axis=1).mean())
    vertex = " ".join(" ".join(f"{value:.9g}" for value in row) for row in nodes)
    segment_mass = REFERENCE_MASS_KG / (NODE_COUNT - 1)
    half_width = geometry["width_m"] * 0.5 + 0.001
    half_thickness = geometry["thickness_m"] * 0.5
    xml = f"""
<mujoco model="strip_cable_structure_probe">
  <option timestep="0.001" gravity="0 0 0"/>
  <extension><plugin plugin="mujoco.elasticity.cable"/></extension>
  <worldbody>
    <body name="strip">
      <freejoint name="strip_free"/>
      <!-- STRUCTURE PROBE ONLY: bend/twist/damping below are non-physical sentinels. -->
      <composite prefix="strip_" type="cable" initial="none" vertex="{vertex}">
        <plugin plugin="mujoco.elasticity.cable">
          <config key="twist" value="{SENTINEL_TWIST:.9g}"/>
          <config key="bend" value="{SENTINEL_BEND:.9g}"/>
        </plugin>
        <joint kind="main" damping="{SENTINEL_DAMPING:.9g}"/>
        <geom type="box"
              size="{0.5 * segment_length:.9g} {half_thickness:.9g} {half_width:.9g}"
              mass="{segment_mass:.12g}" group="3" condim="4"
              solref="0.004 1" solimp="0.95 0.995 0.001"
              friction="1.3 0.02 0.001"/>
        <skin rgba="0.1 0.1 0.11 1" subgrid="3"/>
      </composite>
    </body>
  </worldbody>
</mujoco>
"""
    return xml, geometry, nodes


def compile_structure_probe(obj_path=DEFAULT_OBJ):
    xml, geometry, nodes = build_structure_probe_xml(obj_path)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    info = object_info(model, "strip")

    lo, hi = [], []
    for geom in sorted(info["geoms"]):
        if int(model.geom_type[geom]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            raise RuntimeError("structure probe generated a non-box collision geom")
        half = np.abs(data.geom_xmat[geom].reshape(3, 3)) @ model.geom_size[geom]
        lo.append(data.geom_xpos[geom] - half)
        hi.append(data.geom_xpos[geom] + half)
    cable_lo = np.min(lo, axis=0)
    cable_hi = np.max(hi, axis=0)
    bbox_error_mm = 1000.0 * np.maximum(
        np.abs(cable_lo - geometry["bbox_lo_m"]),
        np.abs(cable_hi - geometry["bbox_hi_m"]),
    )

    checks = {
        "segments_14": len(info["geoms"]) == 14,
        "internal_ball_joints_13": len(info["ball_qpos_adrs"]) == 13,
        "internal_shape_dim_52": 4 * len(info["ball_qpos_adrs"]) == 52,
        "adjacent_exclusions_13": model.nexclude == 13,
        "single_skin": model.nskin == 1,
        "reference_mass": abs(info["mass_kg"] - REFERENCE_MASS_KG) < 1e-9,
        "geometry_within_1p1mm": bool(np.max(bbox_error_mm) <= GEOMETRY_TOLERANCE_MM),
        "finite_initial_state": bool(np.isfinite(model.qpos0).all()),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"cable structure probe failed: {failed}")

    report = {
        "mode": "structure_probe",
        "physical_parameters": False,
        "source_obj": os.path.abspath(obj_path),
        "sentinel": {
            "bend": SENTINEL_BEND,
            "twist": SENTINEL_TWIST,
            "damping": SENTINEL_DAMPING,
            "reference_mass_kg": REFERENCE_MASS_KG,
        },
        "source": {
            "vertices": int(len(geometry["vertices"])),
            "sections": int(len(geometry["section_y"])),
            "bbox_lo_m": geometry["bbox_lo_m"].tolist(),
            "bbox_hi_m": geometry["bbox_hi_m"].tolist(),
            "width_m": geometry["width_m"],
            "thickness_m": geometry["thickness_m"],
            "centerline_arc_m": geometry["centerline_arc_m"],
        },
        "compiled": {
            "object_bodies": len(info["bodies"]),
            "segments": len(info["geoms"]),
            "internal_ball_joints": len(info["ball_qpos_adrs"]),
            "internal_shape_dim": 4 * len(info["ball_qpos_adrs"]),
            "nq": int(model.nq),
            "nv": int(model.nv),
            "skins": int(model.nskin),
            "adjacent_exclusions": int(model.nexclude),
            "mass_kg": info["mass_kg"],
            "local_bbox_lo_m": cable_lo.tolist(),
            "local_bbox_hi_m": cable_hi.tolist(),
            "bbox_abs_error_mm": bbox_error_mm.tolist(),
            "nodes": nodes.tolist(),
        },
        "checks": checks,
    }
    return model, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", default=DEFAULT_OBJ)
    parser.add_argument(
        "--structure-probe",
        action="store_true",
        help="explicitly allow compile-only use of non-physical sentinel parameters",
    )
    args = parser.parse_args(argv)
    if not args.structure_probe:
        raise SystemExit("refusing sentinel parameters without explicit --structure-probe")
    _, report = compile_structure_probe(args.obj)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
