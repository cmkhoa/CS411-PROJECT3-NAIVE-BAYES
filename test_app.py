"""Algorithm checks using abstract tokens, never submission dataset sentences."""

import copy
import csv
import math
import tempfile
import unicodedata
import unittest
from pathlib import Path
from unittest.mock import patch

import app


class NaiveBayesTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        app.model = None
        self.train = self.write("train.csv", [
            ("aaa aaa bbb", "positive"), ("bbb ccc", "negative")
        ])
        app.train_model(self.train)

    def write(self, name, rows):
        path = Path(self.temp.name) / name
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["text", "label"])
            writer.writerows(rows)
        return path

    def test_vietnamese_unicode(self):
        text = "TIẾNG VIỆT! 123"
        expected = ["tiếng", "việt"]
        self.assertEqual(app.tokenize(text), expected)
        self.assertEqual(app.tokenize(unicodedata.normalize("NFD", text)), expected)

    def test_counts_and_laplace_smoothing(self):
        self.assertEqual(app.model["vocabulary"], {"aaa", "bbb", "ccc"})
        self.assertEqual(app.model["totals"], {"positive": 3, "negative": 2})
        self.assertEqual(app.model["priors"], {"positive": 0.5, "negative": 0.5})
        self.assertEqual(app.model["counts"]["positive"]["aaa"], 2)
        self.assertAlmostEqual(app.word_likelihood("aaa", "positive"), 3 / 6)
        self.assertAlmostEqual(app.word_likelihood("aaa", "negative"), 1 / 5)
        for label in app.LABELS:
            self.assertAlmostEqual(sum(app.word_likelihood(word, label)
                                       for word in app.model["vocabulary"]), 1)

    def test_repetition_and_unknown_words(self):
        scores = app.score_text("aaa aaa zzz")
        self.assertAlmostEqual(scores["positive"], math.log(0.5) + 2 * math.log(3 / 6))
        self.assertAlmostEqual(scores["negative"], math.log(0.5) + 2 * math.log(1 / 5))
        self.assertEqual(scores, app.score_text("aaa aaa"))
        self.assertEqual(app.predict("aaa"), "positive")
        self.assertEqual(app.predict("ccc"), "negative")
        self.assertEqual(app.predict("zzz"), "positive")
        self.assertEqual(app.predict(""), "positive")
        self.assertTrue(all(math.isfinite(score)
                            for score in app.score_text("aaa " * 10000).values()))

    def test_ties_and_unequal_priors(self):
        path = self.write("unequal.csv", [
            ("aaa", "positive"), ("bbb", "negative"), ("bbb", "negative")
        ])
        app.train_model(path)
        self.assertEqual(app.predict("zzz"), "negative")
        with patch.object(app, "score_text", return_value=dict.fromkeys(app.LABELS, -2)):
            self.assertEqual(app.predict("aaa"), "negative")
        app.train_model(self.train)
        with patch.object(app, "score_text", return_value=dict.fromkeys(app.LABELS, -2)):
            self.assertEqual(app.predict("aaa"), "positive")

    def test_evaluation_and_no_test_leakage(self):
        path = self.write("test.csv", [
            ("aaa zzz", "positive"), ("ccc", "negative"),
            ("aaa", "negative"), ("ccc", "positive")
        ])
        before = copy.deepcopy(app.model)
        self.assertEqual(app.evaluate_model(path), {
            "accuracy": 0.5, "confusion_matrix": {"TP": 1, "TN": 1, "FP": 1, "FN": 1}
        })
        self.assertEqual(app.model, before)

    def test_retraining_replaces_counts(self):
        changed = self.write("changed.csv", [
            ("ddd", "positive"), ("aaa aaa aaa", "negative")
        ])
        app.train_model(changed)
        self.assertEqual(app.predict("aaa"), "negative")
        self.assertNotIn("ccc", app.model["vocabulary"])
        app.train_model(self.train)
        self.assertEqual(app.predict("aaa"), "positive")

    def test_invalid_data_and_untrained_calls(self):
        for rows in ([], [("aaa", "positive")], [("aaa", "Positive")],
                     [("", "negative")], [("aaa",)],
                     [("aaa", "positive", "extra")],
                     [("123", "positive"), ("!!!", "negative")]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                app.train_model(self.write("invalid.csv", rows))
        with self.assertRaises(RuntimeError):
            app.predict("aaa")
        with self.assertRaises(RuntimeError):
            app.evaluate_model(self.train)
        invalid = Path(self.temp.name) / "invalid.csv"
        invalid.write_text("sentence,label\naaa,positive\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            app.train_model(invalid)


if __name__ == "__main__":
    unittest.main()
