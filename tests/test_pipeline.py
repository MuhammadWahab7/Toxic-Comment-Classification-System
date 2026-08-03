import csv
import json
import tempfile
import unittest
from pathlib import Path

from toxic_comment_classifier.pipeline import load_dataset, run_experiment


NON_TOXIC = [
    "thank you for the helpful explanation",
    "I appreciate your careful review",
    "this discussion is useful and clear",
    "please share the updated document",
    "the solution works well for me",
    "I respectfully disagree with this point",
    "great effort by the entire team",
    "could you explain the final step",
    "this is a thoughtful suggestion",
    "the example made the topic easier",
]

TOXIC = [
    "you are an awful idiot",
    "this is a stupid and useless response",
    "shut up nobody wants your opinion",
    "you are completely worthless",
    "what a pathetic comment",
    "your idea is garbage",
    "only a fool would write this",
    "you are rude and disgusting",
    "stop posting this nonsense",
    "this person is an absolute moron",
]


def write_dataset(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["comment_text", "toxic"])
        writer.writeheader()
        for _ in range(2):
            writer.writerows({"comment_text": text, "toxic": 0} for text in NON_TOXIC)
            writer.writerows({"comment_text": text, "toxic": 1} for text in TOXIC)


class PipelineTests(unittest.TestCase):
    def test_load_dataset_rejects_missing_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid.csv"
            path.write_text("text,label\nhello,0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_dataset(path)

    def test_load_dataset_rejects_fractional_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid-labels.csv"
            path.write_text(
                "comment_text,toxic\nhello,0\nunsafe,0.5\nworld,1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "integer binary labels"):
                load_dataset(path)

    def test_run_experiment_writes_metrics_and_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "sample.csv"
            output = root / "output"
            write_dataset(dataset)

            result = run_experiment(dataset, output, max_features=500)

            self.assertEqual(result["dataset_rows"], 40)
            self.assertTrue((output / "metrics.json").is_file())
            self.assertTrue((output / "toxic_comment_pipeline.joblib").is_file())
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertIn("f1_toxic", metrics)
            self.assertEqual(len(metrics["confusion_matrix"]), 2)


if __name__ == "__main__":
    unittest.main()
