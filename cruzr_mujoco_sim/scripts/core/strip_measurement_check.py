#!/usr/bin/env python3
"""Validate real strip measurements and report the cable/extensible-model gate.

The checker never fits or emits MuJoCo material parameters.
"""

import argparse
import json
import math

import numpy as np


SCHEMA_VERSION = 1
CABLE_LOAD_N = 30.0
CABLE_LOAD_TOLERANCE_N = 0.5
CABLE_EXTENSION_LIMIT_M = 0.005
MIN_TRIALS = 2


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _scalar(errors, path, value, *, positive=False, nonnegative=False):
    if not _number(value):
        errors.append(f"{path} must be a finite number")
        return None
    value = float(value)
    if positive and value <= 0.0:
        errors.append(f"{path} must be > 0")
    if nonnegative and value < 0.0:
        errors.append(f"{path} must be >= 0")
    return value


def _repeats(errors, path, values, *, positive=False, nonnegative=False, minimum=2):
    if not isinstance(values, list) or len(values) < minimum:
        errors.append(f"{path} must contain at least {minimum} repeats")
        return []
    return [
        _scalar(errors, f"{path}[{index}]", value, positive=positive, nonnegative=nonnegative)
        for index, value in enumerate(values)
    ]


def _trials(errors, section, name):
    trials = section.get("trials") if isinstance(section, dict) else None
    if not isinstance(trials, list) or len(trials) < MIN_TRIALS:
        errors.append(f"{name}.trials must contain at least {MIN_TRIALS} repeats")
        return []
    return trials


def _section(errors, document, name):
    section = document.get(name)
    if not isinstance(section, dict):
        errors.append(f"{name} must be an object")
        return {}
    return section


def _ordered_points(errors, path, points, x_name, y_name, minimum_positive):
    if not isinstance(points, list):
        errors.append(f"{path} must be a list")
        return []
    parsed = []
    xs = []
    for index, point in enumerate(points):
        if not isinstance(point, dict):
            errors.append(f"{path}[{index}] must be an object")
            continue
        x = _scalar(errors, f"{path}[{index}].{x_name}", point.get(x_name), nonnegative=True)
        y = _scalar(errors, f"{path}[{index}].{y_name}", point.get(y_name), nonnegative=True)
        if x is not None:
            xs.append(x)
        if x is not None and y is not None:
            parsed.append((x, y, point))
    if xs and any(right <= left for left, right in zip(xs, xs[1:])):
        errors.append(f"{path}.{x_name} must be strictly increasing")
    if not any(abs(value) <= 1e-12 for value in xs):
        errors.append(f"{path} must include a zero-load point")
    if len([value for value in xs if value > 0.0]) < minimum_positive:
        errors.append(f"{path} must include at least {minimum_positive} positive-load points")
    return parsed


def validate_measurements(document):
    errors = []
    warnings = []
    if not isinstance(document, dict):
        return {
            "complete": False,
            "physical_parameters_generated": False,
            "formal_collection_allowed": False,
            "model_decision": "undetermined",
            "errors": ["measurement document must be a JSON object"],
            "warnings": [],
        }
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(document.get("specimen_id"), str) or not document["specimen_id"].strip():
        errors.append("specimen_id must be a non-empty string")
    if document.get("measurement_units") != "field_suffixes_v1":
        errors.append("measurement_units must be field_suffixes_v1; use the unit named in each field")

    provenance = document.get("provenance")
    measured = None
    provenance_kind = "unspecified"
    formal_collection_allowed = True
    if isinstance(provenance, dict):
        measured = provenance.get("measured")
        provenance_kind = str(provenance.get("kind", "unspecified"))
        formal_collection_allowed = provenance.get("formal_collection_allowed") is not False
        if measured is False:
            formal_collection_allowed = False
            warnings.append(
                "synthetic engineering assumptions are valid for model development only; "
                "formal collection requires physical measurements"
            )

    geometry = _section(errors, document, "geometry")
    geometry_values = {}
    for key in ("mass_kg", "length_m", "width_m", "thickness_m", "natural_arch_height_m"):
        geometry_values[key] = _repeats(errors, f"geometry.{key}", geometry.get(key), positive=True)
    geometry_values["balance_offset_m"] = _repeats(
        errors, "geometry.balance_offset_m", geometry.get("balance_offset_m")
    )

    axial = _section(errors, document, "axial_tension")
    axial_error_start = len(errors)
    _scalar(errors, "axial_tension.gauge_length_m", axial.get("gauge_length_m"), positive=True)
    extension_at_30 = []
    for trial_index, trial in enumerate(_trials(errors, axial, "axial_tension")):
        path = f"axial_tension.trials[{trial_index}]"
        if not isinstance(trial, dict):
            errors.append(f"{path} must be an object")
            continue
        points = _ordered_points(errors, f"{path}.points", trial.get("points"), "load_n", "extension_m", 3)
        for load, extension, raw in points:
            hold = _scalar(errors, f"{path}.points@{load:g}N.hold_s", raw.get("hold_s"), nonnegative=True)
            if load > 0.0 and hold is not None and hold < 5.0:
                errors.append(f"{path}.points@{load:g}N.hold_s must be >= 5")
            if abs(load - CABLE_LOAD_N) <= CABLE_LOAD_TOLERANCE_N:
                extension_at_30.append(extension)
        if not any(abs(load - CABLE_LOAD_N) <= CABLE_LOAD_TOLERANCE_N for load, _, _ in points):
            errors.append(f"{path}.points needs a measured load within 30±0.5 N")
        _scalar(errors, f"{path}.residual_extension_m", trial.get("residual_extension_m"), nonnegative=True)
    axial_ready = len(errors) == axial_error_start and len(extension_at_30) >= MIN_TRIALS

    bending = _section(errors, document, "three_point_bending")
    span = _scalar(errors, "three_point_bending.support_span_m", bending.get("support_span_m"), positive=True)
    lengths = [value for value in geometry_values.get("length_m", []) if value is not None]
    if span is not None and lengths and span >= min(lengths):
        errors.append("three_point_bending.support_span_m must be shorter than the specimen")
    for trial_index, trial in enumerate(_trials(errors, bending, "three_point_bending")):
        path = f"three_point_bending.trials[{trial_index}]"
        if not isinstance(trial, dict):
            errors.append(f"{path} must be an object")
            continue
        _ordered_points(errors, f"{path}.points", trial.get("points"), "load_n", "center_deflection_m", 3)
        _scalar(errors, f"{path}.residual_deflection_m", trial.get("residual_deflection_m"), nonnegative=True)

    torsion = _section(errors, document, "torsion")
    _scalar(errors, "torsion.gauge_length_m", torsion.get("gauge_length_m"), positive=True)
    for trial_index, trial in enumerate(_trials(errors, torsion, "torsion")):
        path = f"torsion.trials[{trial_index}]"
        if not isinstance(trial, dict):
            errors.append(f"{path} must be an object")
            continue
        _ordered_points(errors, f"{path}.points", trial.get("points"), "torque_nm", "angle_rad", 2)
        _scalar(errors, f"{path}.residual_angle_rad", trial.get("residual_angle_rad"), nonnegative=True)

    decay = _section(errors, document, "free_decay")
    rate = _scalar(errors, "free_decay.sample_rate_hz", decay.get("sample_rate_hz"), positive=True)
    if rate is not None and rate < 60.0:
        errors.append("free_decay.sample_rate_hz must be >= 60")
    for trial_index, trial in enumerate(_trials(errors, decay, "free_decay")):
        path = f"free_decay.trials[{trial_index}]"
        if not isinstance(trial, dict):
            errors.append(f"{path} must be an object")
            continue
        _scalar(errors, f"{path}.initial_displacement_m", trial.get("initial_displacement_m"), positive=True)
        times = _repeats(errors, f"{path}.peak_time_s", trial.get("peak_time_s"), nonnegative=True, minimum=4)
        peaks = _repeats(
            errors, f"{path}.peak_displacement_m", trial.get("peak_displacement_m"), positive=True, minimum=4
        )
        if len(times) != len(peaks):
            errors.append(f"{path} peak_time_s and peak_displacement_m lengths differ")
        clean_times = [value for value in times if value is not None]
        clean_peaks = [value for value in peaks if value is not None]
        if len(clean_times) == len(times) and any(b <= a for a, b in zip(clean_times, clean_times[1:])):
            errors.append(f"{path}.peak_time_s must be strictly increasing")
        if len(clean_peaks) == len(peaks) and clean_peaks[-1] >= clean_peaks[0]:
            errors.append(f"{path}.peak_displacement_m must show net decay")
        stable = _scalar(errors, f"{path}.stable_time_s", trial.get("stable_time_s"), positive=True)
        if stable is not None and clean_times and stable <= clean_times[-1]:
            errors.append(f"{path}.stable_time_s must be after the last recorded peak")

    friction = _section(errors, document, "friction")
    friction_mu = {}
    for key in ("shelf_critical_angle_deg", "gripper_pad_critical_angle_deg"):
        values = _repeats(errors, f"friction.{key}", friction.get(key), minimum=3)
        clean = [value for value in values if value is not None]
        if any(value <= 0.0 or value >= 90.0 for value in clean):
            errors.append(f"friction.{key} values must be between 0 and 90 degrees")
        if values and len(clean) == len(values):
            friction_mu[key.replace("_critical_angle_deg", "_mu_mean")] = float(
                np.mean(np.tan(np.deg2rad(clean)))
            )

    decision = "undetermined"
    cable_gate = {
        "load_target_n": CABLE_LOAD_N,
        "extension_limit_m": CABLE_EXTENSION_LIMIT_M,
        "worst_measured_extension_m": None,
        "passed": None,
    }
    if axial_ready:
        worst = max(extension_at_30)
        passed = worst <= CABLE_EXTENSION_LIMIT_M
        decision = "cable_candidate" if passed else "requires_extensible_model"
        cable_gate["worst_measured_extension_m"] = worst
        cable_gate["passed"] = passed
    else:
        warnings.append("cable/extensible-model decision remains undetermined until both 30 N trials pass input checks")

    return {
        "complete": not errors,
        "physical_parameters_generated": False,
        "formal_collection_allowed": formal_collection_allowed,
        "measurement_provenance": {
            "kind": provenance_kind,
            "measured": measured,
        },
        "model_decision": decision,
        "cable_gate": cable_gate,
        "friction_summary": friction_mu,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("measurement_json")
    args = parser.parse_args(argv)
    try:
        with open(args.measurement_json, encoding="utf-8") as handle:
            document = json.load(handle)
        report = validate_measurements(document)
    except (OSError, ValueError) as exc:
        report = {
            "complete": False,
            "physical_parameters_generated": False,
            "formal_collection_allowed": False,
            "model_decision": "undetermined",
            "errors": [f"cannot read measurement JSON: {exc}"],
            "warnings": [],
        }
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["complete"] else 1)


if __name__ == "__main__":
    main()
