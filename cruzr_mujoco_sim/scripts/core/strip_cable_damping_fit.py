#!/usr/bin/env python3
"""Fit cable-joint damping with a reproducible isolated decay protocol.

The source data has no fixture protocol, so only its dimensionless logarithmic
decrement is fitted. The source period is reported but is not used as a gate.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import statistics

import mujoco
import numpy as np

from strip_cable_isolated import build_isolated_xml, load_parameter_candidate
from strip_measurement_check import validate_measurements


MODE_BETA = 1.875104068711961
DECREMENT_TOLERANCE = 0.005
PEAK_CONSISTENCY_TOLERANCE = 0.02
PERIOD_RELATIVE_TOLERANCE = 0.02
DECAY_PROTOCOL = {
    "boundary_condition": "full_length_fixed_first_segment",
    "excitation": "analytic_first_cantilever_mode_curvature",
    "bending_axis": "weak_axis_local_z_rotation",
    "observed_body": "strip_B_last",
    "gravity_m_s2": [0.0, 0.0, 0.0],
    "contacts": False,
}


def load_decay_source(path, expected_sha256):
    with open(path, "rb") as handle:
        raw = handle.read()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("measurement SHA-256 does not match parameter candidate")
    document = json.loads(raw.decode("utf-8"))
    validation = validate_measurements(document)
    if not validation["complete"]:
        raise ValueError("decay source is incomplete: " + "; ".join(validation["errors"]))
    displacement = float(statistics.median(
        float(trial["initial_displacement_m"])
        for trial in document["free_decay"]["trials"]
    ))
    return (
        displacement,
        document["free_decay"].get("protocol"),
        actual_sha256,
    )


def first_mode_curvature_weights(count):
    if count < 2:
        raise ValueError("first-mode excitation needs at least two joints")
    position = np.arange(1, count + 1, dtype=np.float64) / (count + 1)
    beta = MODE_BETA
    sigma = (math.cosh(beta) + math.cos(beta)) / (
        math.sinh(beta) + math.sin(beta)
    )
    weights = (
        np.cosh(beta * position)
        + np.cos(beta * position)
        - sigma * (np.sinh(beta * position) + np.sin(beta * position))
    )
    return weights / float(np.max(weights))


def decay_protocol_matches(protocol):
    return isinstance(protocol, dict) and all(
        protocol.get(key) == value for key, value in DECAY_PROTOCOL.items()
    )


def _fixed_model(parameters, damping, obj_path):
    xml, _, _ = build_isolated_xml(
        parameters, obj_path=obj_path, damping=damping, free_root=False
    )
    model = mujoco.MjModel.from_xml_string(xml)
    tip_body = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "strip_B_last"
    )
    if tip_body < 0 or model.njnt != 13:
        raise RuntimeError("fixed-root decay model has unexpected topology")
    return model, tip_body


def _set_first_mode(model, tip_body, target_displacement_m):
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    reference_tip_z = float(data.xpos[tip_body, 2])
    weights = first_mode_curvature_weights(model.njnt)
    axis = np.asarray([0.0, 0.0, 1.0])

    def set_scale(scale):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        for joint, weight in enumerate(weights):
            address = int(model.jnt_qposadr[joint])
            quaternion = np.empty(4)
            mujoco.mju_axisAngle2Quat(quaternion, axis, scale * weight)
            data.qpos[address:address + 4] = quaternion
        mujoco.mj_forward(model, data)
        return abs(float(data.xpos[tip_body, 2]) - reference_tip_z)

    low, high = 0.0, 0.02
    while set_scale(high) < target_displacement_m and high < 0.5:
        high *= 2.0
    if set_scale(high) < target_displacement_m:
        raise RuntimeError("requested decay displacement is outside excitation range")
    for _ in range(48):
        middle = 0.5 * (low + high)
        if set_scale(middle) < target_displacement_m:
            low = middle
        else:
            high = middle
    scale = 0.5 * (low + high)
    actual = set_scale(scale)
    signed = float(data.xpos[tip_body, 2]) - reference_tip_z
    return data, reference_tip_z, scale, actual, signed


def _simulate(parameters, damping, obj_path, displacement_m, duration_s):
    model, tip_body = _fixed_model(parameters, damping, obj_path)
    data, reference_z, scale, actual, signed = _set_first_mode(
        model, tip_body, displacement_m
    )
    samples = [signed]
    finite = True
    for _ in range(int(math.ceil(duration_s / model.opt.timestep))):
        mujoco.mj_step(model, data)
        value = float(data.xpos[tip_body, 2]) - reference_z
        samples.append(value)
        finite = finite and bool(
            math.isfinite(value)
            and np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and np.isfinite(data.qacc).all()
        )
        if not finite:
            break
    warning_count = int(sum(int(item.number) for item in data.warning))
    return {
        "timestep_s": float(model.opt.timestep),
        "samples": np.asarray(samples),
        "finite": finite,
        "warning_count": warning_count,
        "mode_angle_scale_rad": scale,
        "actual_initial_displacement_m": actual,
    }


def _same_side_peaks(response, earliest_s, period_window=None):
    values = response["samples"]
    if values[0] < 0.0:
        indices = np.flatnonzero(
            (values[1:-1] < values[:-2]) & (values[1:-1] <= values[2:])
        ) + 1
    else:
        indices = np.flatnonzero(
            (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])
        ) + 1
    times = indices * response["timestep_s"]
    keep = times >= earliest_s
    if period_window is not None:
        keep &= times >= period_window[0]
        keep &= times <= period_window[1]
    return [
        {
            "time_s": float(index * response["timestep_s"]),
            "amplitude_m": abs(float(values[index])),
        }
        for index in indices[keep]
    ]


def _one_cycle(parameters, damping, obj_path, displacement_m, period_s):
    response = _simulate(
        parameters, damping, obj_path, displacement_m, 1.35 * period_s
    )
    peaks = _same_side_peaks(
        response,
        0.5 * period_s,
        (0.65 * period_s, 1.35 * period_s),
    )
    if not response["finite"] or response["warning_count"] or not peaks:
        return math.inf, None
    decrement = math.log(
        response["actual_initial_displacement_m"] / peaks[0]["amplitude_m"]
    )
    return decrement, peaks[0]


def fit_damping(parameters, *, obj_path, initial_displacement_m,
                source_protocol=None, iterations=16):
    target = float(parameters["decay_target"]["log_decrement"])
    source_period = float(parameters["decay_target"]["peak_period_s"])
    protocol_comparable = decay_protocol_matches(source_protocol)
    if protocol_comparable:
        period = source_period
        valid = []
        candidates = np.geomspace(1e-4, 10.0, 26)[::-1]
        for damping in candidates:
            decrement, peak = _one_cycle(
                parameters, float(damping), obj_path, initial_displacement_m, period
            )
            if peak is not None and math.isfinite(decrement):
                valid.append((float(damping), decrement))
                has_below = any(item[1] <= target for item in valid)
                has_above = any(item[1] >= target for item in valid)
                if has_below and has_above:
                    break
        below = [item for item in valid if item[1] <= target]
        above = [item for item in valid if item[1] >= target]
        if not below or not above:
            raise RuntimeError("target decrement has no stable damping search bracket")
        low, low_decrement = max(below, key=lambda item: item[0])
        high, high_decrement = min(
            (item for item in above if item[0] > low),
            key=lambda item: item[0],
        )
        reference_period_source = "matched_source_protocol"
    else:
        undamped = _simulate(
            parameters, 0.0, obj_path, initial_displacement_m, 20.0
        )
        undamped_peaks = _same_side_peaks(undamped, 0.5)
        if (
            not undamped["finite"]
            or undamped["warning_count"]
            or len(undamped_peaks) < 2
        ):
            raise RuntimeError("could not identify two undamped first-mode peaks")
        period = float(statistics.median(
            right["time_s"] - left["time_s"]
            for left, right in zip(undamped_peaks, undamped_peaks[1:])
        ))
        low, high = 0.0, 0.25
        low_decrement, _ = _one_cycle(
            parameters, low, obj_path, initial_displacement_m, period
        )
        high_decrement, _ = _one_cycle(
            parameters, high, obj_path, initial_displacement_m, period
        )
        while high_decrement < target and high < 2.0:
            high *= 2.0
            high_decrement, _ = _one_cycle(
                parameters, high, obj_path, initial_displacement_m, period
            )
        if not low_decrement <= target <= high_decrement:
            raise RuntimeError("target decrement is outside stable damping search bracket")
        reference_period_source = "undamped_simulation"

    search_high = high
    trace = []
    for _ in range(int(iterations)):
        middle = 0.5 * (low + high)
        decrement, peak = _one_cycle(
            parameters, middle, obj_path, initial_displacement_m, period
        )
        trace.append({
            "damping_nms_per_rad": middle,
            "log_decrement": decrement,
            "first_peak": peak,
        })
        if decrement < target:
            low = middle
        else:
            high = middle
    fitted = 0.5 * (low + high)

    final = _simulate(
        parameters, fitted, obj_path, initial_displacement_m, 2.35 * period
    )
    peaks = _same_side_peaks(final, 0.5 * period)[:2]
    if len(peaks) != 2:
        raise RuntimeError("fitted damping did not produce two measurable peaks")
    decrements = [
        math.log(final["actual_initial_displacement_m"] / peaks[0]["amplitude_m"]),
        math.log(peaks[0]["amplitude_m"] / peaks[1]["amplitude_m"]),
    ]
    observed = float(statistics.median(decrements))
    observed_period = peaks[1]["time_s"] - peaks[0]["time_s"]
    period_relative_error = abs(observed_period - source_period) / source_period
    checks = {
        "finite_response": final["finite"],
        "target_was_bracketed": low_decrement <= target <= high_decrement,
        "initial_displacement_matches": abs(
            final["actual_initial_displacement_m"] - initial_displacement_m
        ) <= 1e-6,
        "log_decrement_matches": abs(observed - target) <= DECREMENT_TOLERANCE,
        "successive_decrements_consistent": abs(decrements[0] - decrements[1])
        <= PEAK_CONSISTENCY_TOLERANCE,
        "period_matches_when_comparable": (
            not protocol_comparable
            or period_relative_error <= PERIOD_RELATIVE_TOLERANCE
        ),
    }
    return {
        "schema_version": 1,
        "mode": "isolated_cable_damping_fit",
        "source_measurement_provenance": parameters["source_measurement_provenance"],
        "formal_collection_allowed": bool(parameters["formal_collection_allowed"]),
        "fit_ready_for_scene_template": all(checks.values()),
        "protocol": {
            "boundary_condition": "full_length_fixed_first_segment",
            "excitation": "analytic_first_cantilever_mode_curvature",
            "bending_axis": "weak_axis_local_z_rotation",
            "observed_body": "strip_B_last",
            "initial_tip_displacement_m": initial_displacement_m,
            "gravity_m_s2": [0.0, 0.0, 0.0],
            "contacts": False,
            "source_protocol_provided": source_protocol is not None,
            "source_protocol_matches": protocol_comparable,
        },
        "fit": {
            "joint_damping_nms_per_rad": fitted,
            "target_log_decrement": target,
            "observed_log_decrement": observed,
            "per_cycle_log_decrement": decrements,
            "reference_period_s": period,
            "reference_period_source": reference_period_source,
            "observed_period_s": observed_period,
            "mode_angle_scale_rad": final["mode_angle_scale_rad"],
            "initial_displacement_actual_m": final["actual_initial_displacement_m"],
            "peaks": peaks,
            "search_initial_bracket_nms_per_rad": [0.0, search_high],
            "iterations": int(iterations),
        },
        "period_comparison": {
            "source_target_period_s": source_period,
            "observed_period_s": observed_period,
            "relative_error": period_relative_error,
            "comparable": protocol_comparable,
            "reason": (
                "source protocol matches the isolated calibration protocol"
                if protocol_comparable else
                "source decay data has no matching fixture/boundary protocol; "
                "period is reported but is not a damping-fit gate"
            ),
        },
        "checks": checks,
        "search_trace": trace,
    }


def calibrated_parameters(parameters, report, parameter_sha, measurement_sha):
    result = copy.deepcopy(parameters)
    result["mode"] = "isolated_cable_parameter_calibrated"
    result["source_parameter_sha256"] = parameter_sha
    result["source_measurement_sha256"] = measurement_sha
    result["fit_ready_for_scene_template"] = report["fit_ready_for_scene_template"]
    result["mujoco"]["joint_damping_nms_per_rad"] = report["fit"][
        "joint_damping_nms_per_rad"
    ]
    result["damping_calibration"] = {
        key: report[key]
        for key in ("protocol", "fit", "period_comparison", "checks")
    }
    return result


def _write_new(path, payload):
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite existing output: {path}")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter_json")
    parser.add_argument("measurement_json")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--report-output")
    parser.add_argument("--calibrated-output")
    parser.add_argument("--iterations", type=int, default=16)
    args = parser.parse_args(argv)

    parameters, parameter_sha = load_parameter_candidate(args.parameter_json)
    displacement, protocol, measurement_sha = load_decay_source(
        args.measurement_json, parameters["source_measurement_sha256"]
    )
    report = fit_damping(
        parameters,
        obj_path=args.obj,
        initial_displacement_m=displacement,
        source_protocol=protocol,
        iterations=args.iterations,
    )
    report["source_parameter_sha256"] = parameter_sha
    report["source_measurement_sha256"] = measurement_sha
    calibrated = calibrated_parameters(
        parameters, report, parameter_sha, measurement_sha
    )
    report_payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    calibrated_payload = json.dumps(calibrated, indent=2, allow_nan=False) + "\n"
    if args.report_output:
        _write_new(args.report_output, report_payload)
    if args.calibrated_output and report["fit_ready_for_scene_template"]:
        _write_new(args.calibrated_output, calibrated_payload)
    print(report_payload, end="")
    raise SystemExit(0 if report["fit_ready_for_scene_template"] else 1)


if __name__ == "__main__":
    main()
