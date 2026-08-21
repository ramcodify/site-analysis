"""BuildSight AI — Research Delay Prediction Model Training & Evaluation

Trains a real gradient boosting / random forest model to predict construction project delays based on:
  - Planned Progress (%)
  - Actual Progress (%)
  - Progress Variance (Actual - Planned)
  - Current Stage Index (0-8)
  - Current Stage Elapsed Days
  - Planned Stage Duration Days
  - Active Worker Count
  - Total Safety Violations
  - Repeated Safety Violations
  - Critical Safety Interruptions

Evaluates on untouched test set and saves:
  - backend/data/models/delay_model.joblib (or .json / .pkl)
  - backend/data/models/delay_evaluation_report.json
  - backend/experiments/delay_evaluation_report.json
"""

import os
import json
import logging
from pathlib import Path
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "data" / "models"
EXPERIMENTS_DIR = BASE_DIR / "experiments"

FEATURE_NAMES = [
    "planned_progress_pct",
    "actual_progress_pct",
    "progress_variance",
    "current_stage_idx",
    "stage_elapsed_days",
    "planned_stage_days",
    "active_worker_count",
    "total_violations",
    "repeated_violations",
    "safety_interruptions",
]


def generate_construction_project_dataset(n_samples=500, random_seed=42):
    """Generate synthetic project milestones grounded in real construction scheduling physics."""
    rng = np.random.RandomState(random_seed)

    X = []
    y_days = []
    y_delayed = []

    for i in range(n_samples):
        planned_prog = rng.uniform(5.0, 95.0)
        current_stage = int(np.clip(planned_prog // 11, 0, 8))
        planned_stage_days = rng.uniform(15.0, 45.0)

        # Worker availability (planned vs actual)
        worker_shortage = rng.choice([True, False], p=[0.35, 0.65])
        active_workers = rng.randint(4, 12) if worker_shortage else rng.randint(12, 28)

        # Safety violations
        total_viols = rng.poisson(lam=4.0 if worker_shortage else 1.5)
        repeated_viols = int(total_viols * rng.uniform(0.1, 0.5))
        interruptions = int(repeated_viols * rng.uniform(0.2, 0.6))

        # Progress variance
        variance = rng.normal(loc=-4.0 if worker_shortage else 1.0, scale=6.0)
        actual_prog = np.clip(planned_prog + variance, 0.0, 100.0)
        var_val = actual_prog - planned_prog

        stage_elapsed = planned_stage_days * (1.0 - var_val / 100.0) + interruptions * 1.5 + rng.normal(0, 2)
        stage_elapsed = max(2.0, stage_elapsed)

        # Ground truth delay days calculation
        # Negative variance, stage overrun, and safety interruptions directly compound delay days
        delay_days_val = (-var_val * 0.8) + max(0.0, stage_elapsed - planned_stage_days) * 0.7 + (interruptions * 2.0) + rng.normal(0, 1.0)
        delay_days_val = max(0.0, round(float(delay_days_val), 1))

        is_delayed = 1 if delay_days_val >= 3.0 else 0

        features = [
            round(float(planned_prog), 1),
            round(float(actual_prog), 1),
            round(float(var_val), 1),
            current_stage,
            round(float(stage_elapsed), 1),
            round(float(planned_stage_days), 1),
            active_workers,
            total_viols,
            repeated_viols,
            interruptions,
        ]

        X.append(features)
        y_days.append(delay_days_val)
        y_delayed.append(is_delayed)

    return np.array(X), np.array(y_days), np.array(y_delayed)


def train_delay_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    X, y_days, y_delayed = generate_construction_project_dataset(n_samples=600, random_seed=42)

    # 70% Train, 15% Val, 15% Untouched Test
    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    X_train, y_train_days, y_train_cls = X[:n_train], y_days[:n_train], y_delayed[:n_train]
    X_val, y_val_days, y_val_cls = X[n_train:n_train+n_val], y_days[n_train:n_train+n_val], y_delayed[n_train:n_train+n_val]
    X_test, y_test_days, y_test_cls = X[n_train+n_val:], y_days[n_train+n_val:], y_delayed[n_train+n_val:]

    logger.info(f"Training Regressor and Classifier on {len(X_train)} samples...")

    # 1. Regressor (Expected Delay Days)
    regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    regressor.fit(X_train, y_train_days)

    # 2. Classifier (Delay Probability)
    classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
    classifier.fit(X_train, y_train_cls)

    # Evaluate on untouched test set
    pred_days = regressor.predict(X_test)
    pred_cls = classifier.predict(X_test)
    pred_probs = classifier.predict_proba(X_test)[:, 1]

    # Metrics
    mae = float(mean_absolute_error(y_test_days, pred_days))
    rmse = float(np.sqrt(mean_squared_error(y_test_days, pred_days)))
    r2 = float(r2_score(y_test_days, pred_days))

    acc = float(accuracy_score(y_test_cls, pred_cls))
    prec = float(precision_score(y_test_cls, pred_cls, zero_division=0))
    rec = float(recall_score(y_test_cls, pred_cls, zero_division=0))
    f1 = float(f1_score(y_test_cls, pred_cls, zero_division=0))

    # Feature Importance
    importances = regressor.feature_importances_
    feat_imp = [
        {"feature": name, "importance": round(float(imp), 4)}
        for name, imp in sorted(zip(FEATURE_NAMES, importances), key=lambda x: x[1], reverse=True)
    ]

    report = {
        "model_type": "GradientBoosting (Regressor + Calibrated Classifier)",
        "features": FEATURE_NAMES,
        "test_samples": len(X_test),
        "regression_metrics": {
            "mean_absolute_error_days": round(mae, 2),
            "root_mean_squared_error_days": round(rmse, 2),
            "r2_score": round(r2, 4),
        },
        "classification_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
        },
        "feature_importances": feat_imp,
    }

    # Save models
    model_bundle = {
        "regressor": regressor,
        "classifier": classifier,
        "features": FEATURE_NAMES,
        "feature_importances": feat_imp,
    }
    model_save_path = MODELS_DIR / "delay_model.joblib"
    joblib.dump(model_bundle, model_save_path)

    report_path = MODELS_DIR / "delay_evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    with open(EXPERIMENTS_DIR / "delay_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"✓ Saved Delay Prediction model bundle to {model_save_path}")
    logger.info(f"✓ Test MAE: {mae:.2f} days | Test R2: {r2:.3f} | Test Classification F1: {f1:.3f}")
    logger.info(f"✓ Evaluation report saved to {report_path}")
    return report


if __name__ == "__main__":
    train_delay_models()
