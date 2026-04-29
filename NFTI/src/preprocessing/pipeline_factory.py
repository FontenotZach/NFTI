"""
Leakage-safe sklearn / imblearn pipelines for tabular NFTI features.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

if TYPE_CHECKING:
    from src.config import TrainingConfig


def build_preprocess_column_transformer(
    binary_cols: List[str],
    categorical_cols: List[str],
    continuous_cols: List[str],
    config: "TrainingConfig",
) -> ColumnTransformer:
    numeric_cols = list(binary_cols) + list(continuous_cols)
    transformers: List[Tuple[str, object, List[str]]] = []

    if numeric_cols:
        num_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=config.numeric_imputer_strategy,
                        add_indicator=config.add_missing_indicators,
                    ),
                ),
            ]
        )
        transformers.append(("num", num_pipe, numeric_cols))

    if categorical_cols:
        cat_pipe = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy=config.categorical_imputer_strategy),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, categorical_cols))

    if not transformers:
        raise ValueError("No feature columns to build ColumnTransformer.")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0)


def build_xgb_classifier_pipeline(
    binary_cols: List[str],
    categorical_cols: List[str],
    continuous_cols: List[str],
    config: "TrainingConfig",
) -> Pipeline | ImbPipeline:
    """
    Full estimator: preprocess -> (optional SMOTE) -> XGBClassifier.

    SMOTE runs only during fit inside CV folds when wrapped by GridSearchCV.
    """
    ct = build_preprocess_column_transformer(
        binary_cols, categorical_cols, continuous_cols, config
    )

    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=config.random_seed,
    )

    if config.use_smote:
        smote = SMOTE(**config.smote_kwargs)
        return ImbPipeline(
            steps=[
                ("preprocess", ct),
                ("smote", smote),
                ("xgb", clf),
            ]
        )

    return Pipeline(
        steps=[
            ("preprocess", ct),
            ("xgb", clf),
        ]
    )
