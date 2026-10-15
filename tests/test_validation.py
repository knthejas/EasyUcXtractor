"""Document-specific reviewed facts must not reject unrelated datasheets."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from easyucxtractor.validation import validate_artifacts


class ValidationScopeTests(unittest.TestCase):
    def test_reviewed_temperature_fact_is_only_required_for_tle9563(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("knowledge.jsonl", "parameters.jsonl", "figures.jsonl"):
                (folder / name).write_text("", encoding="utf-8")
            for title, expected in (("TJA1043", "pass"), ("TC38x", "pass"),
                                    ("TLE9563-3QX", "fail")):
                with self.subTest(title=title):
                    (folder / "output.json").write_text(json.dumps({
                        "texts": [{"text": title}], "pictures": [],
                    }), encoding="utf-8")
                    report = validate_artifacts(
                        folder / "output.json", folder / "knowledge.jsonl",
                        folder / "parameters.jsonl", folder / "figures.jsonl", folder / "report.json",
                    )
                    self.assertEqual(report["status"], expected)


if __name__ == "__main__":
    unittest.main()