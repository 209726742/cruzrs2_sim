#!/usr/bin/env python3
"""Materialize a paired two-roll scene for v16 color counterfactuals."""

from __future__ import annotations

import copy
from pathlib import Path
import xml.etree.ElementTree as ET

from sorting_roll_diversity import APPEARANCE_PROFILES


SUPPORT_NAMES = tuple(
    f"roll_support_x_{side}_{part}_{kind}"
    for side in ("negative", "positive")
    for part in ("base", "robot_lip", "far_lip")
    for kind in ("visual", "col")
)
TABLE_CENTER_X_M = -0.31
TABLE_HALF_WIDTH_X_M = 0.6


def _numbers(value):
    return [float(item) for item in value.split()]


def _format(values):
    return " ".join(f"{value:.9g}" for value in values)


def _named(root, tag, name):
    matches = [item for item in root.iter(tag) if item.get("name") == name]
    if len(matches) != 1:
        raise ValueError(f"scene must contain exactly one {tag} named {name}")
    return matches[0]


def _rename_distractor_body(body, color):
    body.set("name", "sorting_roll_distractor")
    _named(body, "freejoint", "sorting_roll_free").set(
        "name", "sorting_roll_distractor_free"
    )
    visual = _named(body, "geom", "sorting_roll_visual")
    visual.set("name", "sorting_roll_distractor_visual")
    visual.attrib.pop("material", None)
    visual.set("rgba", _format(APPEARANCE_PROFILES[color]["rgba"]))
    collider = _named(body, "geom", "sorting_roll_col")
    collider.set("name", "sorting_roll_distractor_col")
    collider.set("rgba", _format(APPEARANCE_PROFILES[color]["rgba"]))


def materialize_counterfactual_scene(base_scene, destination, assignment):
    """Write one immutable C scene; the visible lane layout stays pair-invariant."""
    if assignment.get("scenario_family") != "C":
        raise ValueError("counterfactual scene requires a C assignment")
    scene = assignment.get("counterfactual_scene") or {}
    target_lane = scene.get("target_lane")
    distractor_lane = scene.get("distractor_lane")
    lane_x = scene.get("lane_x_m") or {}
    lane_colors = scene.get("lane_colors") or {}
    if {target_lane, distractor_lane} != {"left", "right"}:
        raise ValueError("counterfactual assignment must bind left/right lanes")
    if set(lane_x) != {"left", "right"} or set(lane_colors) != {"left", "right"}:
        raise ValueError("counterfactual lane coordinates/colors are incomplete")

    base_scene = Path(base_scene).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite scene: {destination}")
    if destination.parent != base_scene.parent:
        raise ValueError("derived scene must stay beside the base scene for asset paths")

    tree = ET.parse(base_scene)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("scene is missing worldbody")

    table_mesh = _named(root, "mesh", "sorting_table_mesh")
    table_scale = _numbers(table_mesh.get("scale"))
    table_scale[0] = TABLE_HALF_WIDTH_X_M
    table_mesh.set("scale", _format(table_scale))
    table_body = _named(worldbody, "body", "sorting_table")
    table_position = _numbers(table_body.get("pos"))
    table_position[0] = TABLE_CENTER_X_M
    table_body.set("pos", _format(table_position))
    table_top = _named(worldbody, "geom", "table_top_col")
    table_size = _numbers(table_top.get("size"))
    table_size[0] = TABLE_HALF_WIDTH_X_M
    table_top.set("size", _format(table_size))
    table_top_position = _numbers(table_top.get("pos"))
    table_top_position[0] = TABLE_CENTER_X_M
    table_top.set("pos", _format(table_top_position))
    for name in ("table_pedestal_col", "table_base_col"):
        geom = _named(worldbody, "geom", name)
        position = _numbers(geom.get("pos"))
        position[0] += TABLE_CENTER_X_M
        geom.set("pos", _format(position))

    support_clones = []
    for name in SUPPORT_NAMES:
        support = _named(worldbody, "geom", name)
        clone = copy.deepcopy(support)
        clone.set("name", f"distractor_{name}")
        position = _numbers(support.get("pos"))
        target_position = list(position)
        target_position[0] += float(lane_x[target_lane])
        support.set("pos", _format(target_position))
        distractor_position = list(position)
        distractor_position[0] += float(lane_x[distractor_lane])
        clone.set("pos", _format(distractor_position))
        support_clones.append(clone)
    for clone in support_clones:
        worldbody.append(clone)

    target_body = _named(worldbody, "body", "sorting_roll")
    distractor_body = copy.deepcopy(target_body)
    target_position = _numbers(target_body.get("pos"))
    target_position[0] = float(lane_x[target_lane])
    target_body.set("pos", _format(target_position))
    distractor_position = list(target_position)
    distractor_position[0] = float(lane_x[distractor_lane])
    distractor_body.set("pos", _format(distractor_position))
    _rename_distractor_body(distractor_body, lane_colors[distractor_lane])
    worldbody.append(distractor_body)

    destination.parent.mkdir(parents=True, exist_ok=True)
    tree.write(destination, encoding="utf-8", xml_declaration=False)
    return {
        "scene_path": str(destination),
        "target_lane": target_lane,
        "distractor_lane": distractor_lane,
        "lane_x_m": {name: float(value) for name, value in lane_x.items()},
        "lane_colors": dict(lane_colors),
        "target_color": assignment["target_color"],
        "distractor_color": assignment["distractor_color"],
        "support_geom_count_per_lane": len(SUPPORT_NAMES),
        "table_center_x_m": TABLE_CENTER_X_M,
        "table_half_width_x_m": TABLE_HALF_WIDTH_X_M,
        "pair_visible_layout_invariant": True,
        "internal_names_select_target_only": True,
    }
