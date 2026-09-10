import uuid
from datetime import datetime
from typing import Optional
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from app.models.schemas import MLCalibrationBucket, MLTrainResult
from app.services.ml_features import extract_features, FEATURE_NAMES

# Below this many resolved signals on either side of the split, training isn't meaningful --
# a "model" fit on a handful of rows is just memorizing noise, not learning anything a
# forward-looking prediction should be trusted against.
MIN_TRAIN_SIGNALS = 10
MIN_TEST_SIGNALS = 5

# Fixed-width probability ranges for the calibration breakdown, not quantiles -- quantiles
# would always show ~20% of signals in the "top bucket" by construction even if the model
# has zero skill; fixed ranges let a bucket come back empty or tiny, which is itself the
# honest answer when the model isn't confidently separating anything.
CALIBRATION_BUCKETS = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.01)]

# XGBoost over LightGBM: level-wise tree growth and these conservative defaults are the safer
# choice on a small (~hundreds of rows), frequently-retrained dataset without a tuning budget
# -- LightGBM's leaf-wise growth and speed advantage target much larger datasets than this
# classifier ever sees (it already retrains in single-digit milliseconds). Shallow trees
# (MAX_DEPTH), a modest learning rate, and a fixed, modest round count are starting guesses,
# NOT tuned via a search -- same "too small a sample to responsibly tune against" principle
# every other unvalidated constant in this project already follows (see
# consensus.PROXIMITY_ATR_MULT, strategies.SMC_WICK_BODY_MULT).
N_ESTIMATORS = 100
MAX_DEPTH = 3
LEARNING_RATE = 0.1


def _to_xy(signals: list[dict]) -> tuple[list[list[float]], list[int]]:
    X = [[extract_features(s)[name] for name in FEATURE_NAMES] for s in signals]
    # "hit" -> 1; "miss" and "expired" both -> 0 -- both mean the trade didn't actually pay
    # off, which is the real distinction this classifier needs to learn, not just "did it
    # touch the target specifically."
    y = [1 if s["status"] == "hit" else 0 for s in signals]
    return X, y


def _calibration_buckets(probs: list[float], y_test: list[int]) -> list[MLCalibrationBucket]:
    buckets = []
    for low, high in CALIBRATION_BUCKETS:
        bucket_outcomes = [y for p, y in zip(probs, y_test) if low <= p < high]
        if not bucket_outcomes:
            continue  # an empty range is itself informative -- shown as absent, not a fabricated 0%
        label = f"{int(low * 100)}-{min(int(high * 100), 100)}%"
        buckets.append(MLCalibrationBucket(
            range_label=label,
            count=len(bucket_outcomes),
            actual_hit_rate_pct=round(100 * sum(bucket_outcomes) / len(bucket_outcomes), 1),
        ))
    return buckets


def _scale_pos_weight(y_train: list[int]) -> float:
    """
    XGBoost has no class_weight="balanced" knob the way LogisticRegression did -- the
    documented equivalent for binary classification is scale_pos_weight = (# negative /
    # positive), which reweights the positive ("hit") class's gradient the same way
    class_weight="balanced" reweighted LogisticRegression's loss, since hit/miss/expired
    outcomes aren't evenly split (expired folds into the same 0 class as miss, see _to_xy's
    docstring) and the majority class would otherwise dominate the loss and bias
    predict_proba toward it regardless of features. Falls back to 1.0 (no reweighting) for
    the degenerate all-one-class edge case, which can't be meaningfully balanced against
    itself anyway.
    """
    positives = sum(y_train)
    negatives = len(y_train) - positives
    if positives == 0 or negatives == 0:
        return 1.0
    return negatives / positives


def _make_model(y_train: list[int]) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, learning_rate=LEARNING_RATE,
        scale_pos_weight=_scale_pos_weight(y_train), eval_metric="logloss",
    )


def train_hit_classifier(signals: list[dict], train_frac: float = 0.7) -> MLTrainResult:
    """
    Trains an XGBoost classifier to predict hit (1) vs not-hit (0) from each resolved signal's
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

    model = _make_model(y_train)
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
    test_probs = model.predict_proba(X_test)[:, 1].tolist()

    # feature_importances_, NOT signed coefficients -- unlike LogisticRegression's coef_,
    # XGBoost's gain-based importance has no sign or "pushes toward hit vs. away from it"
    # direction, only "how much the model relied on this feature." Reported honestly under
    # that name rather than keeping the old signed-coefficient framing on a model that
    # doesn't have one -- see MLTrainResult.feature_importances' own docstring.
    feature_importances = {
        name: round(float(imp), 5) for name, imp in zip(FEATURE_NAMES, model.feature_importances_.tolist())
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
        feature_importances=feature_importances,
        test_calibration=_calibration_buckets(test_probs, y_test),
    )


def fit_hit_classifier(resolved_signals: list[dict]) -> Optional[XGBClassifier]:
    """
    Fits on ALL given signals, no train/test split -- unlike train_hit_classifier (whose
    entire job is honestly reporting out-of-sample accuracy), a caller wanting a fitted model
    to score new signals against wants all available data used, not a held-out test set.
    Returns None below the same MIN_TRAIN_SIGNALS+MIN_TEST_SIGNALS floor train_hit_classifier
    uses -- "not enough data yet" surfaced as an absent model, not a model fit on noise.

    Factored out of predict_hit_probability (which now just calls this once and predicts)
    so a caller that needs to score MANY inputs against the same fit -- e.g. rl_engine.py's
    compute_ml_scores, scoring every bar of a training walk -- can fit once and reuse the
    model, instead of paying XGBClassifier.fit's cost again per input. predict_proba's shape
    and this "fit once, score many" contract are unchanged from the prior LogisticRegression
    implementation -- rl_engine.py's frozen-snapshot state feature depends on exactly this.
    """
    if len(resolved_signals) < MIN_TRAIN_SIGNALS + MIN_TEST_SIGNALS:
        return None

    X, y = _to_xy(resolved_signals)
    model = _make_model(y)
    model.fit(X, y)
    return model


def predict_hit_probability(resolved_signals: list[dict], new_features: dict[str, float]) -> Optional[float]:
    """
    Best-effort probability for one new, real signal -- see fit_hit_classifier's docstring for
    why this doesn't hold out a test set. Returns None when there isn't enough data to bother
    -- surfaced to the caller as "not enough data yet," not a fabricated number.
    """
    model = fit_hit_classifier(resolved_signals)
    if model is None:
        return None

    new_X = [[new_features[name] for name in FEATURE_NAMES]]
    probability_of_hit = model.predict_proba(new_X)[0][1]
    return round(float(probability_of_hit), 4)
