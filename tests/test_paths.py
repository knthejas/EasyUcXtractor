"""Tests for deterministic, content-specific datasheet output paths."""

import tempfile
import unittest
from pathlib import Path
import sys

ROOT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_PATH / "src"))

from easyucxtractor.paths import ROOT, output_directories


class DatasheetPathTests(unittest.TestCase):
    def test_same_filename_with_different_content_gets_separate_output_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first" / "same.pdf"
            second = Path(directory) / "second" / "same.pdf"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"first datasheet")
            second.write_bytes(b"second datasheet")
            first_dirs = output_directories(first)
            second_dirs = output_directories(second)
        self.assertNotEqual(first_dirs[2], second_dirs[2])
        self.assertEqual(first_dirs[0].parent, ROOT / "data" / "extracted")
        self.assertEqual(first_dirs[1].parent, ROOT / "data" / "knowledge_base")

    def test_id_is_repeatable_for_same_pdf_path_and_content(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "datasheet.pdf"
            first.write_bytes(b"identical contents")
            first_id = output_directories(first)[2]
            second_id = output_directories(first)[2]
        self.assertEqual(first_id, second_id)


if __name__ == "__main__":
    unittest.main()
