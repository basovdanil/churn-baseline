#!/usr/bin/env python3
"""Churn baseline: публичный датасет Telco Customer Churn, CV и подбор порога.

Подкоманды: fetch, train, predict, check. Коды возврата: 0 готово,
2 аргументы, 3 нет файла/сети/модели, 4 ошибка данных или расхождение проверки.
"""

import argparse
import hashlib
import json
import pickle
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
SEED = 42
SOURCE_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
SOURCE_SHA256 = "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
DEFAULT_DATA = ROOT / "data" / "Telco-Customer-Churn.csv"
DEFAULT_MODEL = ROOT / "work" / "model.pkl"
DEFAULT_METRICS = ROOT / "work" / "metrics.json"
DEFAULT_SCORES = ROOT / "work" / "scores.csv"

TARGET = "Churn"
ID = "customerID"
DROP = [ID, TARGET]
SERVICE_COLS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]
CAT_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUM_COLS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "charges_per_month",
    "services_count",
]
MODELS = ("logreg", "forest", "boost")
THRESHOLDS = np.round(np.arange(0.05, 0.96, 0.01), 2)


def add_features(df):
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    tenure = out["tenure"].astype(float).to_numpy()
    total = out["TotalCharges"].fillna(0.0).to_numpy()
    per_month = np.divide(total, tenure, out=np.zeros(len(out)), where=tenure > 0)
    fallback = out["MonthlyCharges"].to_numpy()
    out["charges_per_month"] = np.where(tenure > 0, per_month, fallback)
    out["services_count"] = (out[SERVICE_COLS] == "Yes").sum(axis=1).astype(int)
    return out


def split_xy(df):
    data = add_features(df)
    y = (data[TARGET] == "Yes").astype(int).to_numpy()
    X = data.drop(columns=DROP, errors="ignore")
    return X, y


def make_preprocessor():
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, NUM_COLS),
            ("cat", categorical, CAT_COLS),
        ]
    )


def make_model(name):
    if name == "logreg":
        estimator = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=SEED
        )
    elif name == "forest":
        estimator = RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        )
    elif name == "boost":
        estimator = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.06,
            class_weight="balanced",
            random_state=SEED,
        )
    else:
        raise ValueError(f"unknown model: {name}")
    return Pipeline([("prep", make_preprocessor()), ("model", estimator)])


def best_threshold(y, proba):
    best = (0.5, -1.0)
    for threshold in THRESHOLDS:
        score = f1_score(y, (proba >= threshold).astype(int), zero_division=0)
        if score > best[1]:
            best = (float(threshold), float(score))
    return best


def pr_auc(y, proba):
    precision, recall, _ = precision_recall_curve(y, proba)
    return float(auc(recall, precision))


def cv_scores(X, y, folds):
    result = {}
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    for name in MODELS:
        aucs, prs, accs = [], [], []
        oof = np.zeros(len(y))
        for train_idx, test_idx in splitter.split(X, y):
            model = make_model(name).fit(X.iloc[train_idx], y[train_idx])
            proba = model.predict_proba(X.iloc[test_idx])[:, 1]
            oof[test_idx] = proba
            aucs.append(roc_auc_score(y[test_idx], proba))
            prs.append(pr_auc(y[test_idx], proba))
            accs.append(accuracy_score(y[test_idx], (proba >= 0.5).astype(int)))
        threshold, f1 = best_threshold(y, oof)
        result[name] = {
            "roc_auc": float(np.mean(aucs)),
            "roc_auc_std": float(np.std(aucs)),
            "pr_auc": float(np.mean(prs)),
            "pr_auc_std": float(np.std(prs)),
            "accuracy": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "threshold": threshold,
            "f1": f1,
        }
    return result


def load_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"нет файла: {path}")
    df = pd.read_csv(path)
    missing = [c for c in [TARGET, ID] + NUM_COLS[:4] + CAT_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"нет колонок: {', '.join(missing)}")
    return df


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(model, fh)


def load_model(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"нет модели: {path}")
    with path.open("rb") as fh:
        return pickle.load(fh)


def cmd_fetch(args):
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()
    digest = hashlib.sha256(payload).hexdigest()
    if args.expect_sha256 and digest != args.expect_sha256:
        raise ValueError(f"sha256 не совпал: {digest}")
    target.write_bytes(payload)
    df = pd.read_csv(target)
    rate = (df[TARGET] == "Yes").mean()
    print(f"rows={len(df)} bytes={len(payload)} churn={rate:.4f} sha256={digest}")
    return 0


def cmd_train(args):
    df = load_csv(args.data)
    X, y = split_xy(df)
    scores = cv_scores(X, y, args.folds)
    for name in MODELS:
        stats = scores[name]
        print(
            f"{name:7s} auc={stats['roc_auc']:.4f} ±{stats['roc_auc_std']:.4f} "
            f"pr={stats['pr_auc']:.4f} acc={stats['accuracy']:.4f} "
            f"thr={stats['threshold']:.2f} f1={stats['f1']:.4f}"
        )
    best = max(MODELS, key=lambda name: scores[name]["roc_auc"])
    model = make_model(best).fit(X, y)
    save_model(model, args.model)
    payload = {
        "dataset": "Telco Customer Churn",
        "rows": int(len(df)),
        "positives": int(y.sum()),
        "folds": args.folds,
        "best": best,
        "threshold": scores[best]["threshold"],
        "models": scores,
    }
    metrics = Path(args.metrics)
    metrics.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = scores[best]
    print(
        f"BEST={best} auc={stats['roc_auc']:.4f} ±{stats['roc_auc_std']:.4f} "
        f"thr={stats['threshold']:.2f} f1={stats['f1']:.4f} rows={len(df)} folds={args.folds}"
    )
    return 0


def cmd_predict(args):
    df = load_csv(args.data)
    X, y = split_xy(df)
    metrics_path = Path(args.metrics)
    threshold = 0.5
    if metrics_path.exists():
        threshold = float(json.loads(metrics_path.read_text())["threshold"])
    if args.holdout > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=args.holdout,
            stratify=y,
            random_state=SEED,
        )
        model = make_model(args.model).fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        frame = df.loc[X_test.index, [ID]].copy()
        truth = y_test
    else:
        model = load_model(args.model_path)
        proba = model.predict_proba(X)[:, 1]
        frame = df[[ID]].copy()
        truth = y
    frame["churn_probability"] = np.round(proba, 6)
    frame["churn_pred"] = (proba >= threshold).astype(int)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    line = f"rows={len(frame)} flagged={int(frame['churn_pred'].sum())} thr={threshold:.2f}"
    if args.holdout > 0:
        line += f" ACC={accuracy_score(truth, frame['churn_pred']):.4f}"
        line += f" AUC={roc_auc_score(truth, frame['churn_probability']):.4f}"
    print(line)
    return 0


def cmd_check(args):
    checks = []
    df = load_csv(args.data)
    X, y = split_xy(df)
    checks.append(("features", all(c in X.columns for c in NUM_COLS + CAT_COLS)))
    checks.append(("no_target_leak", TARGET not in X.columns and len(X) == len(df)))
    metrics_path = Path(args.metrics)
    payload = json.loads(metrics_path.read_text())
    checks.append(
        (
            "metrics",
            payload.get("best") in MODELS
            and 0.5 <= float(payload.get("models", {}).get(payload["best"], {}).get("roc_auc", 0)) <= 1.0
            and 0.0 < float(payload.get("threshold", 0)) < 1.0,
        )
    )
    model = load_model(args.model_path)
    proba = model.predict_proba(X.head(5))[:, 1]
    checks.append(("model", bool(np.all((proba >= 0) & (proba <= 1)))))
    scores_path = Path(args.scores)
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        checks.append(
            (
                "scores",
                len(scores) == len(df) and {"churn_probability", "churn_pred"} <= set(scores.columns),
            )
        )
        checks.append(
            (
                "scores_range",
                bool(scores["churn_probability"].between(0, 1).all())
                and bool(
                    np.array_equal(
                        scores["churn_pred"].to_numpy(),
                        (scores["churn_probability"] >= payload["threshold"]).astype(int),
                    )
                ),
            )
        )
    else:
        checks.append(("scores", False))
        checks.append(("scores_range", False))
    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"CHECK {name}={'OK' if ok else 'FAIL'}")
    print(f"CHECKED={len(checks)} FAILED={len(failed)}")
    return 4 if failed else 0


def build_parser():
    parser = argparse.ArgumentParser(description="Churn baseline: CV, порог, артефакты")
    subs = parser.add_subparsers(dest="cmd", required=True)

    fetch = subs.add_parser("fetch", help="скачать датасет и проверить sha256")
    fetch.add_argument("--out", default=str(DEFAULT_DATA))
    fetch.add_argument("--expect-sha256", default=SOURCE_SHA256)
    fetch.set_defaults(func=cmd_fetch)

    train = subs.add_parser("train", help="CV по трём моделям и обучение лучшей")
    train.add_argument("--data", default=str(DEFAULT_DATA))
    train.add_argument("--folds", type=int, default=5)
    train.add_argument("--model", default=str(DEFAULT_MODEL))
    train.add_argument("--metrics", default=str(DEFAULT_METRICS))
    train.set_defaults(func=cmd_train)

    predict = subs.add_parser("predict", help="вероятности и бинарный прогноз")
    predict.add_argument("--data", default=str(DEFAULT_DATA))
    predict.add_argument("--model", default="logreg")
    predict.add_argument("--model-path", default=str(DEFAULT_MODEL))
    predict.add_argument("--metrics", default=str(DEFAULT_METRICS))
    predict.add_argument("--holdout", type=float, default=0.0)
    predict.add_argument("--out", default=str(DEFAULT_SCORES))
    predict.set_defaults(func=cmd_predict)

    check = subs.add_parser("check", help="проверить артефакты и согласованность")
    check.add_argument("--data", default=str(DEFAULT_DATA))
    check.add_argument("--model-path", default=str(DEFAULT_MODEL))
    check.add_argument("--metrics", default=str(DEFAULT_METRICS))
    check.add_argument("--scores", default=str(DEFAULT_SCORES))
    check.set_defaults(func=cmd_check)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"ERROR {exc}")
        return 3
    except (ValueError, KeyError) as exc:
        print(f"ERROR {exc}")
        return 4


if __name__ == "__main__":
    sys.exit(main())
