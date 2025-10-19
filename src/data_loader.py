"""Utilities for loading and preprocessing the crop recommendation dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd

_NUMERIC_COLUMNS: Iterable[str] = (
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
)

_COLUMN_ALIASES: Dict[str, str] = {
    "n": "N",
    "nitrogen": "N",
    "p": "P",
    "phosphorus": "P",
    "k": "K",
    "potassium": "K",
    "temperature": "temperature",
    "humidity": "humidity",
    "ph": "ph",
    "rainfall": "rainfall",
    "label": "label",
}


def load_dataset(path: str = "data/crop_recommendation.csv") -> pd.DataFrame:
    """Load the crop recommendation dataset from ``path``.

    Parameters
    ----------
    path : str, optional
        Relative or absolute path to the CSV file, by default ``data/crop_recommendation.csv``.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the raw dataset.

    Raises
    ------
    FileNotFoundError
        If the CSV file cannot be found.
    ValueError
        If the file is empty or cannot be parsed by pandas.
    """

    file_path = Path(path)
    fallback_paths: List[Path] = []

    if not file_path.exists():
        if Path(path) == Path("data/crop_recommendation.csv"):
            fallback_paths.append(Path("data/Crop_recommendation.csv"))
        for candidate in fallback_paths:
            if candidate.exists():
                file_path = candidate
                break
        else:
            raise FileNotFoundError(f"Dataset not found at {file_path.resolve()}")

    try:
        dataframe = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:  # pragma: no cover - defensive coding
        raise ValueError(f"Dataset at {file_path.resolve()} is empty") from exc
    except pd.errors.ParserError as exc:  # pragma: no cover - defensive coding
        raise ValueError(f"Dataset at {file_path.resolve()} could not be parsed") from exc

    return _standardize_columns(dataframe)


def preprocess_data(dataframe: pd.DataFrame, return_scaler: bool = False) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Clean and normalize the dataset.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Raw dataset, typically returned by :func:`load_dataset`.
    return_scaler : bool, optional
        When ``True`` returns a tuple of the processed dataframe and the
        min/max scaling factors used during normalization.

    Returns
    -------
    pandas.DataFrame | tuple
        Preprocessed dataframe with missing values handled and numeric columns normalized.
        If ``return_scaler`` is ``True`` a tuple of *(dataframe, scaling_dict)* is returned.

    Raises
    ------
    ValueError
        If expected numeric columns are missing.
    """

    dataframe = _standardize_columns(dataframe)

    missing_columns = [column for column in _NUMERIC_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {', '.join(missing_columns)}")

    processed = dataframe.copy()
    processed = processed.dropna(subset=_NUMERIC_COLUMNS)

    for column in _NUMERIC_COLUMNS:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")

    processed = processed.dropna(subset=_NUMERIC_COLUMNS)

    scaling_params: Dict[str, Tuple[float, float]] = {}

    for column in _NUMERIC_COLUMNS:
        column_min = processed[column].min()
        column_max = processed[column].max()
        scaling_params[column] = (column_min, column_max)
        if column_min == column_max:
            processed[column] = 0.0
            continue
        processed[column] = (processed[column] - column_min) / (column_max - column_min)

    processed = processed.reset_index(drop=True)

    if return_scaler:
        return processed, scaling_params

    return processed


def _standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe with normalized column names."""

    rename_map = {}
    for column in dataframe.columns:
        normalized = column.strip().lower()
        renamed = _COLUMN_ALIASES.get(normalized, column.strip())
        rename_map[column] = renamed

    standardized = dataframe.rename(columns=rename_map)
    drop_candidates = [col for col in standardized.columns if col.lower().startswith("unnamed")]
    standardized = standardized.drop(columns=drop_candidates, errors="ignore")

    return standardized


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    df_raw = load_dataset()
    print("Raw sample:")
    print(df_raw.head())

    df_preprocessed = preprocess_data(df_raw)
    print("\nPreprocessed sample:")
    print(df_preprocessed.head())
