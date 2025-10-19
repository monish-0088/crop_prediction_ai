"""Model training pipeline for the Crop Prediction AI system."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.data_loader import load_dataset, preprocess_data

LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    """Configure root logging for consistent console output."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def prepare_data() -> Tuple[np.ndarray, np.ndarray, LabelEncoder, Tuple[str, ...], Dict[str, Tuple[float, float]]]:
    """Load, preprocess, and encode the dataset for training.

    Returns
    -------
    tuple
        Feature matrix, encoded labels, fitted label encoder, and the feature names in order.

    Raises
    ------
    ValueError
        If the expected target column ``label`` is missing after preprocessing.
    """

    raw_df = load_dataset()
    processed_df, scaling = preprocess_data(raw_df, return_scaler=True)

    if "label" not in processed_df.columns:
        raise ValueError("Target column 'label' not present in dataset after preprocessing.")

    feature_columns = tuple(col for col in processed_df.columns if col != "label")
    X = processed_df.loc[:, feature_columns].to_numpy(dtype=np.float32)
    y_raw = processed_df["label"].to_numpy()

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y_raw)

    return X, y_encoded, encoder, feature_columns, scaling


def build_model_registry(n_classes: int, random_state: int = 42) -> Dict[str, object]:
    """Return a registry of candidate classifiers."""

    return {
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=random_state),
        "xgboost": XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            objective="multi:softmax",
            eval_metric="mlogloss",
            use_label_encoder=False,
            num_class=n_classes,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            objective="multiclass",
            num_class=n_classes,
        ),
    }


def train_and_evaluate_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> Tuple[Dict[str, float], str, object]:
    """Train candidate models and compute accuracy on the test split."""

    accuracies: Dict[str, float] = {}
    n_classes = int(np.unique(y_train).size)
    models = build_model_registry(n_classes=n_classes, random_state=random_state)
    trained_models: Dict[str, object] = {}

    for name, model in models.items():
        LOGGER.info("Training model: %s", name)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        LOGGER.info("Accuracy for %s: %.4f", name, accuracy)
        accuracies[name] = accuracy
        trained_models[name] = model

    best_model_name = max(accuracies, key=accuracies.get)
    best_model = trained_models[best_model_name]

    return accuracies, best_model_name, best_model


def persist_model(
    model: object,
    encoder: LabelEncoder,
    feature_columns: Tuple[str, ...],
    scaling: Dict[str, Tuple[float, float]],
    output_path: Path = Path("src/models/best_model.pkl"),
) -> Path:
    """Persist the trained model artifacts to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "label_encoder": encoder,
    "feature_columns": feature_columns,
    "scaling": scaling,
    }
    joblib.dump(artifact, output_path)
    LOGGER.info("Saved best model to %s", output_path)
    return output_path


def main() -> None:
    """Entrypoint for training the crop recommendation models."""

    configure_logging()
    LOGGER.info("Preparing dataset")
    X, y, encoder, feature_columns, scaling = prepare_data()

    LOGGER.info("Splitting data train/test")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    LOGGER.info("Training candidate models")
    accuracies, best_model_name, best_model_obj = train_and_evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    best_accuracy = accuracies[best_model_name]

    LOGGER.info("Best model: %s (accuracy %.4f)", best_model_name, best_accuracy)
    persist_model(best_model_obj, encoder, feature_columns, scaling)

    print("Model accuracies:")
    for model_name, accuracy in sorted(accuracies.items()):
        print(f"  {model_name}: {accuracy:.4f}")
    print(f"Best model saved: {best_model_name} (accuracy {best_accuracy:.4f})")


if __name__ == "__main__":
    main()
