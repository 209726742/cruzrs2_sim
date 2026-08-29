#!/usr/bin/env python3

from pathlib import Path
import unittest


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = (
    PACKAGE_ROOT / "scripts" / "collection"
    / "sorting_roll_v16_expansion_collect.sh"
)
TMUX = (
    PACKAGE_ROOT / "scripts" / "collection"
    / "sorting_roll_v16_expansion_tmux.sh"
)


class SortingRollV16ExpansionCollectTests(unittest.TestCase):
    def test_collector_is_bounded_and_resumable(self):
        text = COLLECTOR.read_text(encoding="utf-8")
        self.assertIn("PIDS=()", text)
        self.assertIn("-eq ${#GPUS[@]}", text)
        self.assertIn("already successful; skip", text)
        self.assertIn("--require-complete", text)
        self.assertIn("sorting_roll_v16_expansion_validate.py", text)

    def test_tmux_launcher_detaches_and_logs(self):
        text = TMUX.read_text(encoding="utf-8")
        self.assertIn("tmux new-session -d", text)
        self.assertIn("tmux has-session", text)
        self.assertIn("SORTING_ROLL_V16_GPUS=", text)
        self.assertIn("SORTING_ROLL_V16_EXPANSION_CAMPAIGN=", text)
        self.assertIn("SORTING_ROLL_V16_EXPANSION_OUTPUT_ROOT=", text)
        self.assertIn("SORTING_ROLL_V16_EXPANSION_MANIFEST=", text)
        self.assertIn("sorting_roll_v16_expansion_collect.sh", text)


if __name__ == "__main__":
    unittest.main()
