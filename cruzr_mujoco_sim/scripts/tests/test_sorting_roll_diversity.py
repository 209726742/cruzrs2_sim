#!/usr/bin/env python3

import copy
import sys
from pathlib import Path
import unittest

import numpy as np


CORE_DIR = Path(__file__).resolve().parents[1] / "core"
sys.path.insert(0, str(CORE_DIR))

import sorting_roll_scene as scene  # noqa: E402
from sorting_roll_diversity import (  # noqa: E402
    ADMISSION_GROUPS,
    DIVERSE_TASK_VERSION,
    OBJECT_PROFILES,
    apply_model_diversity,
    generate_manifest,
    manifest_errors,
)
from sorting_roll_task import target_placement_smoke  # noqa: E402


class SortingRollDiversityTest(unittest.TestCase):
    def test_long_profile_preserves_the_24_mm_collision_baseline(self):
        self.assertEqual(OBJECT_PROFILES["long_baseline"]["length_m"], 0.5)
        self.assertEqual(OBJECT_PROFILES["long_baseline"]["diameter_m"], 0.024)

    def test_formal_300_manifest_has_exact_stratified_quotas(self):
        self.assertEqual(DIVERSE_TASK_VERSION, "sorting_roll_v15_diverse_sim")
        manifest = generate_manifest("formal300_contract", 1000, 300)
        self.assertEqual(manifest["task_version"], DIVERSE_TASK_VERSION)
        self.assertEqual(manifest["counts"], {
            "split": {"test": 30, "train": 240, "val": 30},
            "pose_bin": {"boundary": 60, "easy": 120, "medium": 120},
            "prompt_id": {
                "prompt_0": 60,
                "prompt_1": 60,
                "prompt_2": 60,
                "prompt_3": 60,
                "prompt_4": 60,
            },
            "object_profile": {
                "long_baseline": 100,
                "medium": 100,
                "short_slim": 100,
            },
            "appearance_profile": {
                "blue": 60,
                "green": 60,
                "orange": 60,
                "red": 60,
                "yellow": 60,
            },
            "lighting_profile": {
                "bright": 60,
                "dim": 60,
                "normal": 180,
            },
            "dynamics_profile": {
                "heavy_low_friction": 60,
                "light_high_friction": 60,
                "nominal": 180,
            },
            "image_profile": {
                "clean": 180,
                "mild_compression": 60,
                "strong_compression": 60,
            },
        })
        self.assertEqual(
            manifest,
            generate_manifest("formal300_contract", 1000, 300),
        )

    def test_each_admission_manifest_forces_one_physical_profile(self):
        for index, (group, forced) in enumerate(ADMISSION_GROUPS.items()):
            with self.subTest(group=group):
                manifest = generate_manifest(
                    f"admission_{group}",
                    10000 + 20 * index,
                    20,
                    group,
                )
                self.assertEqual(
                    manifest["counts"]["split"],
                    {"test": 2, "train": 16, "val": 2},
                )
                for field, expected in forced.items():
                    self.assertEqual(
                        {item[field]["name"] for item in manifest["assignments"]},
                        {expected},
                    )
                self.assertEqual(
                    set(manifest["counts"]["pose_bin"]),
                    {"easy", "medium", "boundary"},
                )
                self.assertEqual(len(manifest["counts"]["prompt_id"]), 5)
                self.assertEqual(
                    len(manifest["counts"]["appearance_profile"]), 5
                )
                self.assertEqual(
                    len(manifest["counts"]["lighting_profile"]), 3
                )
                self.assertEqual(
                    len(manifest["counts"]["image_profile"]), 3
                )

    def test_tampered_or_malformed_manifest_is_rejected_without_crashing(self):
        manifest = generate_manifest("tamper_contract", 12000, 2)
        tampered = copy.deepcopy(manifest)
        tampered["assignments"][0]["object_profile"]["length_m"] += 0.01
        errors = manifest_errors(tampered)
        self.assertTrue(any("object_profile is invalid" in error for error in errors))
        self.assertTrue(any("assignment_id mismatch" in error for error in errors))

        malformed = copy.deepcopy(manifest)
        malformed["assignments"][0] = None
        errors = manifest_errors(malformed)
        self.assertTrue(any("assignment is not an object" in error for error in errors))

    def test_all_admission_physics_profiles_fit_and_settle_in_target(self):
        import mujoco

        scene_path = scene.materialize_scene()
        for index, group in enumerate(ADMISSION_GROUPS):
            with self.subTest(group=group):
                assignment = generate_manifest(
                    f"physics_{group}",
                    13000 + index,
                    1,
                    group,
                )["assignments"][0]
                model = mujoco.MjModel.from_xml_path(str(scene_path))
                report = apply_model_diversity(
                    mujoco,
                    model,
                    mujoco.MjData(model),
                    assignment,
                )
                collider = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    "sorting_roll_col",
                )
                body = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    "sorting_roll",
                )
                visual = mujoco.mj_name2id(
                    model,
                    mujoco.mjtObj.mjOBJ_GEOM,
                    "sorting_roll_visual",
                )
                profile = assignment["object_profile"]
                dynamics = assignment["dynamics_profile"]
                self.assertAlmostEqual(
                    model.geom_size[collider, 0],
                    profile["diameter_m"] / 2.0,
                )
                self.assertAlmostEqual(
                    model.geom_size[collider, 1],
                    profile["length_m"] / 2.0,
                )
                self.assertAlmostEqual(model.body_mass[body], dynamics["mass_kg"])
                self.assertAlmostEqual(
                    model.geom_friction[collider, 0],
                    dynamics["sliding_friction"],
                )
                material = int(model.geom_matid[visual])
                self.assertTrue(np.all(model.mat_texid[material] == -1))
                np.testing.assert_allclose(
                    model.mat_rgba[material],
                    assignment["appearance_profile"]["rgba"],
                )
                self.assertTrue(report["visual_texture_disabled"])
                spans = np.asarray(report["visual_mesh_span_m"])
                axis = report["visual_length_axis"]
                self.assertAlmostEqual(
                    spans[axis], profile["length_m"], delta=2e-4
                )
                np.testing.assert_allclose(
                    np.delete(spans, axis),
                    profile["diameter_m"],
                    atol=2e-4,
                )
                _, evidence = target_placement_smoke(model, steps=2500)
                self.assertTrue(evidence["success"], evidence)
                self.assertGreaterEqual(
                    min(evidence["endpoint_margin_m"].values()), 0.0
                )


if __name__ == "__main__":
    unittest.main()
