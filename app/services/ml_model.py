import uuid
from datetime import datetime
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from app.models.schemas import MLTrainResult
from app.services.ml_features import extract_features, FEATURE_NAMES

# Below this many resolved signals on either side of the split, training isn't meaningful --
# a "model" fit on a handful of rows is just memorizing noise, not learning anything a
# forward-looking prediction should be trusted against.
MIN_TRAIN_SIGNALS = 10
MIN_TEST_SIGNALS = 5


def _to_xy(signals: list[dict]) -> tuple[list[list[float]], list[int]]:
    X = [[extract_features(s)[name] for name in FEATURE_NAMES] for s in signals]
    # "hit" -> 1; "miss" and "expired" both -> 0 -- both mean the trade didn't actually pay
    # off, which is the real distinction this classifier needs to learn, not just "did it
    # touch the target specifically."
    y = [1 if s["status"] == "hit" else 0 for s in signals]
    return X, y


def train_hit_classifier(signals: list[dict], train_frac: float = 0.7) -> MLTrainResult:
    """
    Trains a LogisticRegression to predict hit (1) vs not-hit (0) from each resolved signal's
    extract_features() output. Splits chronologically by timestamp, NOT randomly -- the same
    lookahead-bias discipline run_backtest's eval_start_index already enforces elsewhere in
    this project. A random shuffle would let the model train on signals that happened AFTER
    some of its test signals, leaking future information into training and producing a
    falsely optimistic test score that wouldn't hold up in real, forward-only use.
    """
    ordered = sorted(signals, key=lambda s: s["timestamp"])
    split_idx = int(len(ordered) * train_frac)
    train_signals = ordered[:split_idx]
    test_signals = ordered[split_idx:]

    if len(train_signals) < MIN_TRAIN_SIGNALS or len(test_signals) < MIN_TEST_SIGNALS:
        raise ValueError(
            f"Not enough resolved signals for a meaningful split: {len(train_signals)} train, "
            f"{len(test_signals)} test (need {MIN_TRAIN_SIGNALS}+/{MIN_TEST_SIGNALS}+). "
            f"Check /signals/accuracy for the current resolved count."
        )

    X_train, y_train = _to_xy(train_signals)
    X_test, y_test = _to_xy(test_signals)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_predictions = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    # zero_division=0 -- if the model predicts "hit" for nobody (or everybody) in the test
    # slice, precision/recall have an undefined denominator; report 0.0 rather than raising,
    # since that's itself informative (the model isn't distinguishing anything yet), not an
    # error to hide.
    test_precision = precision_score(y_test, test_predictions, zero_division=0)
    test_recall = recall_score(y_test, test_predictions, zero_division=0)

    feature_coefficients = {
        name: round(float(coef), 5) for name, coef in zip(FEATURE_NAMES, model.coef_[0].tolist())
    }

    return MLTrainResult(
        run_id=uuid.uuid4().hex[:12],
        created_at=datetime.utcnow(),
        train_samples=len(train_signals),
        test_samples=len(test_signals),
        train_accuracy=round(float(train_accuracy), 4),
        test_accuracy=round(float(test_accuracy), 4),
        test_precision=round(float(test_precision), 4),
        test_recall=round(float(test_recall), 4),
        feature_coefficients=feature_coefficients,
    )


def predict_hit_probability(resolved_signals: list[dict], new_features: dict[str, float]) -> Optional[float]:
    """
    Trains fresh on every resolved signal, no train/test split -- unlike train_hit_classifier
    (whose entire job is honestly reporting out-of-sample accuracy), this is a best-effort
    probability for one new, real signal, so using all available data matters more than
    holding out a test set. Returns None when there isn't enough data to bother -- surfaced
    to the caller as "not enough data yet," not a fabricated number.
    """
    if len(resolved_signals) < MIN_TRAIN_SIGNALS + MIN_TEST_SIGNALS:
        return None

    X, y = _to_xy(resolved_signals)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    new_X = [[new_features[name] for name in FEATURE_NAMES]]
    probability_of_hit = model.predict_proba(new_X)[0][1]
    return round(float(probability_of_hit), 4)
