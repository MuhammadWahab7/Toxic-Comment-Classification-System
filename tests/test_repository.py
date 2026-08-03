import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class RepositoryTests(unittest.TestCase):
    def test_notebook_is_valid_and_has_no_stale_outputs(self):
        path = ROOT / "notebooks" / "toxic_comment_classification.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 20)
        code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertTrue(all(not cell.get("outputs") for cell in code_cells))

        source = "\n".join("".join(cell.get("source", [])) for cell in code_cells)
        for marker in (
            "drop_sibling_label_columns",
            "SMOTE",
            "RandomizedSearchCV",
            "VotingClassifier",
            "StackingClassifier",
        ):
            self.assertIn(marker, source)

    def test_dataset_and_generated_models_are_ignored(self):
        ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("dataset.csv", ignore)
        self.assertIn("*.joblib", ignore)
        self.assertIn("artifacts/", ignore)

    def test_historical_report_is_preserved(self):
        report = ROOT / "docs" / "historical-project-report.txt"
        self.assertTrue(report.is_file())
        self.assertIn("Toxic Comment Classification", report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
