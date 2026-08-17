#!/usr/bin/env python3
"""Fit auditable MuJoCo cable candidates from validated strip measurements.

This tool performs deterministic mechanics conversions only.  It does not tune
parameters against task success and it does not fit joint damping; damping is
left for an isolated free-decay simulation.
"""

import argparse
import hashlib
import json
import math
import os
import statistics

from strip_cable_structure import DEFAULT_OBJ, REFERENCE_MASS_KG, load_strip_geometry
from strip_measurement_check import validate_measurements


SCHEMA_VERSION = 1
GEOMETRY_RELATIVE_TOLERANCE = 0.02
ARCH_ABSOLUTE_TOLERANCE_M = 0.002
MODULUS_RELATIVE_TOLERANCE = 0.15


def _mean(values):
    return float(statistics.fmean(float(value) for value in values))


def _median(values):
    values = [float(value) for value in values]
    if not values:
        raise ValueError("fit needs at least one positive measurement")
    return float(statistics.median(values))


def rectangle_torsion_constant(width_m, thickness_m):
    """Match the rectangular-section formula used by MuJoCo's cable plugin."""
    half_width = 0.5 * float(width_m)
    half_thickness = 0.5 * float(thickness_m)
    a = max(half_width, half_thickness)
    b = min(half_width, half_thickness)
    return float(a * b**3 * (16.0 / 3.0 - 3.36 * b / a * (1.0 - b**4 / a**4 / 12.0)))


def _relative_error(value, reference):
    return abs(float(value) - float(reference)) / abs(float(reference))


def _decay_targets(trials):
    periods = []
    decrements = []
    stable_times = []
    for trial in trials:
        times = [float(value) for value in trial["peak_time_s"]]
        peaks = [float(value) for value in trial["peak_displacement_m"]]
        periods.extend(right - left for left, right in zip(times, times[1:]))
        decrements.extend(
            math.log(left / right) for left, right in zip(peaks, peaks[1:])
        )
        stable_times.append(float(trial["stable_time_s"]))
    period = _median(periods)
    decrement = _median(decrements)
    damping_ratio = decrement / math.sqrt((2.0 * math.pi) ** 2 + decrement**2)
    damped_frequency = 2.0 * math.pi / period
    natural_frequency = damped_frequency / math.sqrt(1.0 - damping_ratio**2)
    return {
        "peak_period_s": period,
        "log_decrement": decrement,
        "damping_ratio": damping_ratio,
        "natural_frequency_rad_s": natural_frequency,
        "stable_time_s": _median(stable_times),
    }


def fit_material(document, *, measurement_sha256=None, obj_path=DEFAULT_OBJ):
    validation = validate_measurements(document)
    if not validation["complete"]:
        raise ValueError("measurement input is incomplete: " + "; ".join(validation["errors"]))
    if validation["model_decision"] != "cable_candidate":
        raise ValueError(
            "material fit requires cable_candidate, got "
            f"{validation['model_decision']}"
        )

    geometry = document["geometry"]
    mass = _mean(geometry["mass_kg"])
    length = _mean(geometry["length_m"])
    width = _mean(geometry["width_m"])
    thickness = _mean(geometry["thickness_m"])
    arch_height = _mean(geometry["natural_arch_height_m"])
    area = width * thickness
    weak_axis_second_moment = width * thickness**3 / 12.0
    strong_axis_second_moment = thickness * width**3 / 12.0
    torsion_constant = rectangle_torsion_constant(width, thickness)

    axial_moduli = []
    gauge_length = float(document["axial_tension"]["gauge_length_m"])
    for trial in document["axial_tension"]["trials"]:
        for point in trial["points"]:
            load = float(point["load_n"])
            extension = float(point["extension_m"])
            if load > 0.0 and extension > 0.0:
                axial_moduli.append(load * gauge_length / (area * extension))

    bending_moduli = []
    support_span = float(document["three_point_bending"]["support_span_m"])
    for trial in document["three_point_bending"]["trials"]:
        for point in trial["points"]:
            load = float(point["load_n"])
            deflection = float(point["center_deflection_m"])
            if load > 0.0 and deflection > 0.0:
                bending_moduli.append(
                    load * support_span**3
                    / (48.0 * weak_axis_second_moment * deflection)
                )

    twist_moduli = []
    torsion_gauge = float(document["torsion"]["gauge_length_m"])
    for trial in document["torsion"]["trials"]:
        for point in trial["points"]:
            torque = float(point["torque_nm"])
            angle = float(point["angle_rad"])
            if torque > 0.0 and angle > 0.0:
                twist_moduli.append(torque * torsion_gauge / (torsion_constant * angle))

    axial_pa = _median(axial_moduli)
    bend_pa = _median(bending_moduli)
    twist_pa = _median(twist_moduli)
    modulus_difference = abs(axial_pa - bend_pa) / (0.5 * (axial_pa + bend_pa))
    implied_poisson = bend_pa / (2.0 * twist_pa) - 1.0

    source = load_strip_geometry(obj_path)
    source_arch = float(
        source["centres"][:, 2].max()
        - 0.5 * (source["centres"][0, 2] + source["centres"][-1, 2])
    )
    geometry_errors = {
        "mass_relative": _relative_error(mass, REFERENCE_MASS_KG),
        "length_relative": _relative_error(length, source["centerline_arc_m"]),
        "width_relative": _relative_error(width, source["width_m"]),
        "thickness_relative": _relative_error(thickness, source["thickness_m"]),
        "arch_absolute_m": abs(arch_height - source_arch),
    }
    gates = {
        "cable_extension": validation["cable_gate"]["passed"] is True,
        "mass_matches_obj_reference": geometry_errors["mass_relative"] <= GEOMETRY_RELATIVE_TOLERANCE,
        "length_matches_obj": geometry_errors["length_relative"] <= GEOMETRY_RELATIVE_TOLERANCE,
        "width_matches_obj": geometry_errors["width_relative"] <= GEOMETRY_RELATIVE_TOLERANCE,
        "thickness_matches_obj": geometry_errors["thickness_relative"] <= GEOMETRY_RELATIVE_TOLERANCE,
        "arch_matches_obj": geometry_errors["arch_absolute_m"] <= ARCH_ABSOLUTE_TOLERANCE_M,
        "axial_bending_consistent": modulus_difference <= MODULUS_RELATIVE_TOLERANCE,
        "isotropic_poisson_plausible": 0.0 <= implied_poisson < 0.5,
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "isolated_cable_parameter_candidate",
        "source_measurement_sha256": measurement_sha256,
        "source_measurement_provenance": validation["measurement_provenance"],
        "formal_collection_allowed": validation["formal_collection_allowed"],
        "parameter_candidate_generated": True,
        "fit_ready_for_isolated_dynamics": all(gates.values()),
        "model_decision": validation["model_decision"],
        "geometry": {
            "mass_kg": mass,
            "length_m": length,
            "width_m": width,
            "thickness_m": thickness,
            "natural_arch_height_m": arch_height,
            "area_m2": area,
            "weak_axis_second_moment_m4": weak_axis_second_moment,
            "strong_axis_second_moment_m4": strong_axis_second_moment,
            "torsion_constant_m4": torsion_constant,
        },
        "elastic_fit": {
            "axial_modulus_pa": axial_pa,
            "bend_pa": bend_pa,
            "twist_pa": twist_pa,
            "axial_bending_relative_difference": modulus_difference,
            "implied_poisson_ratio": implied_poisson,
            "bending_orientation": "flatwise_weak_axis",
        },
        "decay_target": _decay_targets(document["free_decay"]["trials"]),
        "contact": validation["friction_summary"],
        "geometry_errors": geometry_errors,
        "gates": gates,
        "warnings": list(validation["warnings"]),
        "mujoco": {
            "plugin": "mujoco.elasticity.cable",
            "bend_config_pa": bend_pa,
            "twist_config_pa": twist_pa,
            "flat": False,
            "joint_damping_nms_per_rad": None,
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement_json")
    parser.add_argument("--obj", default=DEFAULT_OBJ)
    parser.add_argument("--output")
    args = parser.parse_args(argv)

    with open(args.measurement_json, "rb") as handle:
        raw = handle.read()
    document = json.loads(raw.decode("utf-8"))
    report = fit_material(
        document,
        measurement_sha256=hashlib.sha256(raw).hexdigest(),
        obj_path=args.obj,
    )
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        if os.path.exists(args.output):
            raise SystemExit(f"refusing to overwrite existing output: {args.output}")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(payload)
    print(payload, end="")
    raise SystemExit(0 if report["fit_ready_for_isolated_dynamics"] else 1)


if __name__ == "__main__":
    main()
