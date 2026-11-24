import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import churn

SMALL = ROOT / "tests" / "data" / "telco_small.csv"


def make_row(**overrides):
    row = {
        "customerID": "0001-TEST",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 10,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": "500.0",
        "Churn": "No",
    }
    row.update(overrides)
    return row


class FeaturesTest(unittest.TestCase):
    def test_charges_per_month_is_total_over_tenure(self):
        frame = churn.add_features(pd.DataFrame([make_row()]))
        self.assertAlmostEqual(frame["charges_per_month"].iloc[0], 50.0)

    def test_zero_tenure_falls_back_to_monthly(self):
        frame = churn.add_features(
            pd.DataFrame([make_row(tenure=0, TotalCharges=" ", MonthlyCharges=29.85)])
        )
        self.assertAlmostEqual(frame["charges_per_month"].iloc[0], 29.85)

    def test_blank_total_becomes_nan(self):
        frame = churn.add_features(pd.DataFrame([make_row(TotalCharges="")]))
        self.assertTrue(np.isnan(frame["TotalCharges"].iloc[0]))

    def test_services_count_only_yes(self):
        frame = churn.add_features(pd.DataFrame([make_row()]))
        self.assertEqual(int(frame["services_count"].iloc[0]), 3)

    def test_target_maps_yes_to_one(self):
        frame = pd.DataFrame([make_row(), make_row(Churn="Yes")])
        _, y = churn.split_xy(frame)
        self.assertEqual(int(y.sum()), 1)

    def test_id_and_target_dropped(self):
        frame = pd.DataFrame([make_row()])
        X, _ = churn.split_xy(frame)
        self.assertNotIn("customerID", X.columns)
        self.assertNotIn("Churn", X.columns)
        self.assertTrue(set(churn.NUM_COLS + churn.CAT_COLS) <= set(X.columns))


class ModelTest(unittest.TestCase):
    def test_three_models_are_pipelines(self):
        for name in churn.MODELS:
            with self.subTest(model=name):
                self.assertIn("model", churn.make_model(name).named_steps)

    def test_proba_shape_on_small_data(self):
        X, y = churn.split_xy(pd.read_csv(SMALL))
        proba = churn.make_model("logreg").fit(X, y).predict_proba(X)[:, 1]
        self.assertEqual(proba.shape, (len(X),))
        self.assertTrue(((proba >= 0) & (proba <= 1)).all())


class ThresholdTest(unittest.TestCase):
    def test_best_threshold_on_separable_scores(self):
        y = np.array([0, 0, 1, 1])
        threshold, score = churn.best_threshold(y, np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertAlmostEqual(score, 1.0)
        self.assertTrue(0.2 < threshold <= 0.8)

    def test_pr_auc_perfect_ranking(self):
        y = np.array([0, 0, 1, 1])
        self.assertAlmostEqual(churn.pr_auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0)

    def test_pr_auc_rewards_better_ranking(self):
        y = np.array([0, 0, 0, 1])
        good = np.array([0.1, 0.2, 0.3, 0.9])
        bad = np.array([0.9, 0.3, 0.2, 0.1])
        self.assertGreater(churn.pr_auc(y, good), churn.pr_auc(y, bad))


class CvTest(unittest.TestCase):
    def test_cv_scores_cover_all_models(self):
        X, y = churn.split_xy(pd.read_csv(SMALL))
        scores = churn.cv_scores(X, y, folds=2)
        self.assertEqual(set(scores), set(churn.MODELS))
        for name, stats in scores.items():
            with self.subTest(model=name):
                self.assertTrue(0.5 <= stats["roc_auc"] <= 1.0)
                self.assertTrue(0.0 < stats["threshold"] < 1.0)
                self.assertTrue(0.0 <= stats["f1"] <= 1.0)


class CliTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.model = self.tmp / "model.pkl"
        self.metrics = self.tmp / "metrics.json"
        self.scores = self.tmp / "scores.csv"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, *argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = churn.main(list(argv))
        return code, buffer.getvalue()

    def train(self):
        return self.run_cli(
            "train",
            "--data",
            str(SMALL),
            "--folds",
            "2",
            "--model",
            str(self.model),
            "--metrics",
            str(self.metrics),
        )

    def test_train_writes_metrics(self):
        code, out = self.train()
        self.assertEqual(code, 0)
        self.assertIn("BEST=", out)
        payload = json.loads(self.metrics.read_text(encoding="utf-8"))
        self.assertIn(payload["best"], churn.MODELS)
        self.assertEqual(payload["rows"], len(pd.read_csv(SMALL)))
        self.assertIn("threshold", payload)

    def test_predict_holdout_writes_scores(self):
        self.train()
        code, _ = self.run_cli(
            "predict",
            "--data",
            str(SMALL),
            "--holdout",
            "0.25",
            "--model",
            "logreg",
            "--model-path",
            str(self.model),
            "--metrics",
            str(self.metrics),
            "--out",
            str(self.scores),
        )
        self.assertEqual(code, 0)
        scores = pd.read_csv(self.scores)
        self.assertEqual(len(scores), 30)
        self.assertTrue({"customerID", "churn_probability", "churn_pred"} <= set(scores.columns))

    def test_check_passes_on_trained_artifacts(self):
        self.train()
        self.run_cli(
            "predict",
            "--data",
            str(SMALL),
            "--model-path",
            str(self.model),
            "--metrics",
            str(self.metrics),
            "--out",
            str(self.scores),
        )
        code, out = self.run_cli(
            "check",
            "--data",
            str(SMALL),
            "--model-path",
            str(self.model),
            "--metrics",
            str(self.metrics),
            "--scores",
            str(self.scores),
        )
        self.assertEqual(code, 0)
        self.assertIn("FAILED=0", out)

    def test_missing_data_returns_3(self):
        code, out = self.run_cli("train", "--data", str(self.tmp / "nope.csv"))
        self.assertEqual(code, 3)
        self.assertIn("ERROR", out)

    def test_check_without_model_returns_3(self):
        self.train()
        code, _ = self.run_cli(
            "check",
            "--data",
            str(SMALL),
            "--model-path",
            str(self.tmp / "absent.pkl"),
            "--metrics",
            str(self.metrics),
            "--scores",
            str(self.scores),
        )
        self.assertEqual(code, 3)

    def test_tampered_scores_return_4(self):
        self.train()
        self.run_cli(
            "predict",
            "--data",
            str(SMALL),
            "--model-path",
            str(self.model),
            "--metrics",
            str(self.metrics),
            "--out",
            str(self.scores),
        )
        scores = pd.read_csv(self.scores)
        scores["churn_pred"] = 1 - scores["churn_pred"]
        scores.to_csv(self.scores, index=False)
        code, out = self.run_cli(
            "check",
            "--data",
            str(SMALL),
            "--model-path",
            str(self.model),
            "--metrics",
            str(self.metrics),
            "--scores",
            str(self.scores),
        )
        self.assertEqual(code, 4)
        self.assertIn("scores_range=FAIL", out)

    def test_unknown_model_returns_4(self):
        self.train()
        code, _ = self.run_cli(
            "predict",
            "--data",
            str(SMALL),
            "--holdout",
            "0.2",
            "--model",
            "nope",
            "--model-path",
            str(self.model),
            "--metrics",
            str(self.metrics),
            "--out",
            str(self.scores),
        )
        self.assertEqual(code, 4)


if __name__ == "__main__":
    unittest.main()
