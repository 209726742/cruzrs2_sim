#!/usr/bin/env python3

import subprocess
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
COLLECTION_DIR = PACKAGE_ROOT / "scripts" / "collection"
ADMISSION = COLLECTION_DIR / "sorting_roll_v15_admission.sh"
H100X2_ADMISSION = COLLECTION_DIR / "sorting_roll_v15_h100x2_admission.sh"


class SortingRollV15AdmissionTest(unittest.TestCase):
    def test_shell_entrypoints_have_valid_syntax(self):
        for script in (ADMISSION, H100X2_ADMISSION):
            subprocess.run(["bash", "-n", str(script)], check=True)

    def test_admission_is_pinned_to_v15_camera_and_task(self):
        source = ADMISSION.read_text(encoding="utf-8")
        self.assertIn("sorting_roll_d405_candidate_v6", source)
        self.assertIn("sorting_roll_v15_diverse_sim", source)
        self.assertIn("--finalize-only", source)

    def test_two_h100_workers_cover_each_group_once(self):
        source = H100X2_ADMISSION.read_text(encoding="utf-8")
        self.assertIn("run_worker 0", source)
        self.assertIn("run_worker 1", source)
        self.assertIn("--finalize-only", source)
        for group in (
            "dynamics_heavy_low_friction",
            "dynamics_light_high_friction",
            "geometry_long",
            "geometry_medium",
            "geometry_short",
        ):
            self.assertEqual(source.count(group), 1, group)


if __name__ == "__main__":
    unittest.main()
