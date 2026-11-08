# ai_data_analysis.py
# AI Data Analysis — AI-Powered Automated Analytics and Insight Platform
# Run: streamlit run ai_data_analysis.py

import io
import json
import os
import re
import warnings
from dotenv import load_dotenv

load_dotenv()
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# Optional dependencies
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import simpleSplit
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_CSS = """
<style>
    .main > div { padding-top: 1.2rem; }
    .block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1500px; }
    h1, h2, h3 { letter-spacing: -0.02em; }
    .app-hero {
        background: linear-gradient(120deg, #0f172a 0%, #1e293b 55%, #155e75 100%);
        color: white; padding: 1.7rem 2rem; border-radius: 16px;
        margin-bottom: 1.4rem; border: 1px solid rgba(255,255,255,0.08);
    }
    .app-hero h1 { margin: 0 0 .45rem 0; font-size: 2.1rem; }
    .app-hero p { margin: 0; opacity: .88; font-size: 1rem; }
    .metric-card {
        background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: .85rem 1rem; height: 100%;
    }
    .metric-card .label { color:#64748b; font-size:.82rem; font-weight:600; }
    .metric-card .value { color:#0f172a; font-size:1.35rem; font-weight:700; }
    .quality-good { color:#15803d; font-weight:700; }
    .quality-medium { color:#b45309; font-weight:700; }
    .quality-poor { color:#b91c1c; font-weight:700; }
    .insight-box {
        border-left: 4px solid #0ea5e9; background:#f0f9ff;
        padding: .9rem 1rem; border-radius: 8px; margin: .7rem 0;
    }
    .warning-box {
        border-left: 4px solid #f59e0b; background:#fffbeb;
        padding: .9rem 1rem; border-radius: 8px; margin: .7rem 0;
    }
    .chat-user {
        background:#e0f2fe; border-radius:14px; padding:.75rem 1rem;
        margin:.45rem 0; color:#0c4a6e;
    }
    .chat-ai {
        background:#f1f5f9; border-radius:14px; padding:.75rem 1rem;
        margin:.45rem 0; color:#0f172a;
    }
    .small-muted { color:#64748b; font-size:.86rem; }
    div[data-testid="stDataFrame"] { border: 1px solid #e2e8f0; border-radius: 10px; }
</style>
"""

st.markdown(APP_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_session():
    defaults = {
        "file_name": None,
        "df_original": None,
        "df": None,
        "cleaning_log": [],
        "chat_history": [],
        "analysis_context": {},
        "last_result": None,
        "last_chart": None,
        "groq_status": "unknown",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session()


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or (
            isinstance(value, float) and np.isnan(value)
        ):
            return None

        return float(value)
    except Exception:
        return None


def format_number(value: Any) -> str:
    value = safe_float(value)

    if value is None:
        return "—"

    abs_value = abs(value)

    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"

    if abs_value >= 1_000:
        return f"{value / 1_000:.2f}K"

    if abs_value >= 1:
        return f"{value:,.2f}".rstrip("0").rstrip(".")

    return f"{value:.4f}".rstrip("0").rstrip(".")


def infer_column_types(df: pd.DataFrame) -> Dict[str, str]:
    types = {}

    for col in df.columns:
        series = df[col]

        if pd.api.types.is_bool_dtype(series):
            types[col] = "boolean"

        elif pd.api.types.is_datetime64_any_dtype(series):
            types[col] = "datetime"

        elif pd.api.types.is_numeric_dtype(series):
            types[col] = "numeric"

        else:
            parsed_dates = pd.to_datetime(
                series, errors="coerce", format="mixed"
            )

            date_ratio = (
                parsed_dates.notna().mean()
                if len(series) else 0
            )

            if date_ratio > 0.75 and series.notna().any():
                types[col] = "datetime"
            else:
                types[col] = "categorical"

    return types


def unique_count(series: pd.Series) -> int:
    return int(series.nunique(dropna=True))


def make_json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [make_json_safe(v) for v in obj]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)

    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()

    if isinstance(obj, float) and np.isnan(obj):
        return None

    return obj


def add_cleaning_log(
    action: str,
    details: str,
    before: Any = None,
    after: Any = None
):
    st.session_state.cleaning_log.append({
        "time": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "action": action,
        "details": details,
        "before": make_json_safe(before),
        "after": make_json_safe(after),
    })


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Reading and validating dataset...")
def load_dataframe(
    file_bytes: bytes,
    file_name: str
) -> pd.DataFrame:
    suffix = os.path.splitext(
        file_name.lower()
    )[1]

    if suffix not in [".csv", ".xlsx", ".xls"]:
        raise ValueError(
            "Unsupported file type. Upload a CSV, XLSX, or XLS file."
        )

    if suffix == ".csv":
        errors = []

        for encoding in [
            "utf-8", "utf-8-sig", "latin1", "cp1252"
        ]:
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=encoding,
                    low_memory=False,
                    on_bad_lines="skip",
                )

                if not df.empty:
                    return df

            except UnicodeDecodeError:
                errors.append(
                    f"Encoding {encoding} failed."
                )

            except Exception as exc:
                errors.append(
                    f"Encoding {encoding}: {exc}"
                )

        raise ValueError(
            "Could not parse CSV. "
            + " | ".join(errors[:3])
        )

    try:
        return pd.read_excel(io.BytesIO(file_bytes))

    except Exception as exc:
        raise ValueError(
            f"Could not read Excel file: {exc}"
        )


def normalize_columns(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    df = df.copy()
    notes = []
    original_columns = list(df.columns)
    new_columns = []
    seen = {}

    for col in original_columns:
        name = str(col).strip()

        if not name or name.lower() == "unnamed: 0":
            name = f"column_{len(new_columns) + 1}"

            notes.append(
                f"Renamed unnamed column to '{name}'."
            )

        base = name
        counter = 2

        while name.lower() in seen:
            name = f"{base}_{counter}"
            counter += 1

        if name != str(col):
            notes.append(
                f"Renamed duplicate or invalid column "
                f"'{col}' to '{name}'."
            )

        seen[name.lower()] = True
        new_columns.append(name)

    df.columns = new_columns
    return df, notes


def validate_dataframe(
    df: pd.DataFrame
) -> List[str]:
    issues = []

    if df is None:
        return ["No dataset loaded."]

    if df.empty:
        return [
            "The uploaded dataset contains no data rows."
        ]

    if len(df.columns) == 0:
        return [
            "The uploaded dataset contains no columns."
        ]

    if len(df) > 500_000:
        issues.append(
            "Large dataset detected. Some operations may take longer."
        )

    all_null = [
        c for c in df.columns if df[c].isna().all()
    ]

    if all_null:
        issues.append(
            f"These columns contain only missing values: "
            f"{', '.join(all_null[:10])}."
        )

    return issues


# ---------------------------------------------------------------------------
# Profiling and quality assessment
# ---------------------------------------------------------------------------

def profile_dataframe(
    df: pd.DataFrame
) -> Dict[str, Any]:
    types = infer_column_types(df)
    columns = []

    for col in df.columns:
        series = df[col]
        col_type = types[col]
        missing = int(series.isna().sum())
        unique = unique_count(series)

        profile = {
            "column": col,
            "type": col_type,
            "missing": missing,
            "missing_pct": (
                round(
                    missing / len(df) * 100, 2
                ) if len(df) else 0
            ),
            "unique": unique,
            "duplicates_in_column": int(
                series.duplicated().sum()
            ),
        }

        if col_type == "numeric":
            values = pd.to_numeric(
                series, errors="coerce"
            ).dropna()

            profile.update({
                "mean": (
                    safe_float(values.mean())
                    if len(values) else None
                ),
                "median": (
                    safe_float(values.median())
                    if len(values) else None
                ),
                "std": (
                    safe_float(values.std())
                    if len(values) else None
                ),
                "min": (
                    safe_float(values.min())
                    if len(values) else None
                ),
                "max": (
                    safe_float(values.max())
                    if len(values) else None
                ),
                "zeros": (
                    int((values == 0).sum())
                    if len(values) else 0
                ),
                "negative_values": (
                    int((values < 0).sum())
                    if len(values) else 0
                ),
            })

            if (
                len(values) > 2
                and values.std()
                and values.std() > 0
            ):
                profile["skewness"] = safe_float(
                    values.skew()
                )

            else:
                profile["skewness"] = None

        elif col_type == "datetime":
            parsed = pd.to_datetime(
                series, errors="coerce"
            ).dropna()

            profile.update({
                "min_date": (
                    str(parsed.min())
                    if len(parsed) else None
                ),
                "max_date": (
                    str(parsed.max())
                    if len(parsed) else None
                ),
                "invalid_dates": int(
                    (
                        pd.to_datetime(
                            series, errors="coerce"
                        ).isna() & series.notna()
                    ).sum()
                ),
            })

        else:
            top = series.value_counts(
                dropna=True
            ).head(5)

            profile.update({
                "top_values": {
                    str(k): int(v)
                    for k, v in top.items()
                },
                "high_cardinality": unique > 100,
            })

        columns.append(profile)

    duplicate_rows = int(df.duplicated().sum())

    constant_columns = [
        c for c in df.columns
        if unique_count(df[c]) <= 1
    ]

    high_cardinality = [
        c for c, t in types.items()
        if t == "categorical"
        and unique_count(df[c]) > 100
    ]

    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": duplicate_rows,
        "total_missing": int(
            df.isna().sum().sum()
        ),
        "column_profiles": columns,
        "constant_columns": constant_columns,
        "high_cardinality_columns": high_cardinality,
        "types": types,
    }


def calculate_quality_score(
    df: pd.DataFrame,
    profile: Dict[str, Any]
) -> Tuple[float, List[str]]:
    if df.empty:
        return 0.0, ["Dataset is empty."]

    score = 100.0
    reasons = []

    missing_ratio = (
        profile["total_missing"]
        / (
            profile["rows"] * profile["columns"]
        )
        if profile["rows"] and profile["columns"]
        else 0
    )

    penalty = missing_ratio * 30
    score -= penalty

    if missing_ratio > 0:
        reasons.append(
            f"Missing values reduce the score by "
            f"{penalty:.1f} points."
        )

    duplicate_ratio = (
        profile["duplicate_rows"] / profile["rows"]
        if profile["rows"] else 0
    )

    penalty = duplicate_ratio * 20
    score -= penalty

    if duplicate_ratio > 0:
        reasons.append(
            f"Duplicate rows reduce the score by "
            f"{penalty:.1f} points."
        )

    constant_penalty = min(
        len(profile["constant_columns"]) * 3, 15
    )
    score -= constant_penalty

    if constant_penalty:
        reasons.append(
            f"{len(profile['constant_columns'])} constant or "
            f"near-constant columns reduce the score by "
            f"{constant_penalty} points."
        )

    high_card_penalty = min(
        len(profile["high_cardinality_columns"]) * 2, 10
    )
    score -= high_card_penalty

    if high_card_penalty:
        reasons.append(
            f"{len(profile['high_cardinality_columns'])} "
            f"high-cardinality categorical columns reduce the "
            f"score by {high_card_penalty} points."
        )

    object_numeric = []

    for col in df.columns:
        if profile["types"][col] == "categorical":
            sample = (
                df[col]
                .dropna()
                .astype(str)
                .head(200)
            )

            numeric_like = sample.apply(
                lambda value: bool(
                    re.match(
                        r"^-?\d+(\.\d+)?$",
                        str(value).strip()
                    )
                )
            ).mean()

            if numeric_like > 0.8:
                object_numeric.append(col)

    if object_numeric:
        penalty = min(
            len(object_numeric) * 4, 15
        )
        score -= penalty

        reasons.append(
            f"Possible numeric values stored as text: "
            f"{', '.join(object_numeric[:5])}."
        )

    score = max(0.0, min(100.0, score))

    if not reasons:
        reasons.append(
            "No major quality issues detected by the "
            "built-in checks."
        )

    return round(score, 1), reasons


def quality_label(score: float) -> str:
    if score >= 85:
        return (
            '<span class="quality-good">Good</span>'
        )

    if score >= 65:
        return (
            '<span class="quality-medium">'
            'Needs attention</span>'
        )

    return (
        '<span class="quality-poor">Poor</span>'
    )


# ---------------------------------------------------------------------------
# Cleaning operations
# ---------------------------------------------------------------------------

def remove_duplicate_rows(
    df: pd.DataFrame
) -> pd.DataFrame:
    before = len(df)
    cleaned = df.drop_duplicates().copy()
    removed = before - len(cleaned)

    add_cleaning_log(
        "Remove duplicate rows",
        f"Removed {removed} duplicate rows.",
        before,
        len(cleaned)
    )

    return cleaned


def impute_missing_values(
    df: pd.DataFrame,
    column: str,
    method: str,
    custom_value: Any = None
) -> pd.DataFrame:
    df = df.copy()
    before_missing = int(df[column].isna().sum())

    if method == "Mean":
        value = df[column].mean()

    elif method == "Median":
        value = df[column].median()

    elif method == "Mode":
        mode = df[column].mode(dropna=True)
        value = (
            mode.iloc[0] if len(mode) else None
        )

    elif method == "Custom value":
        value = custom_value

    else:
        value = None

    if value is not None:
        df[column] = df[column].fillna(value)

    after_missing = int(df[column].isna().sum())

    add_cleaning_log(
        "Impute missing values",
        f"Column '{column}': {before_missing} missing "
        f"values handled using {method}.",
        before_missing,
        after_missing
    )

    return df


def drop_missing_rows(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    before = len(df)
    cleaned = df.dropna(subset=[column]).copy()
    removed = before - len(cleaned)

    add_cleaning_log(
        "Drop missing rows",
        f"Removed {removed} rows with missing "
        f"'{column}'.",
        before,
        len(cleaned)
    )

    return cleaned


def convert_column_type(
    df: pd.DataFrame,
    column: str,
    target_type: str
) -> pd.DataFrame:
    df = df.copy()

    try:
        if target_type == "Numeric":
            df[column] = pd.to_numeric(
                df[column], errors="coerce"
            )

        elif target_type == "Datetime":
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                format="mixed"
            )

        elif target_type == "Categorical":
            df[column] = df[column].astype(str)

        elif target_type == "Boolean":
            df[column] = df[column].astype(bool)

        add_cleaning_log(
            "Convert column type",
            f"Converted '{column}' to {target_type}."
        )

    except Exception as exc:
        st.error(
            f"Could not convert '{column}': {exc}"
        )

    return df


def standardize_text(
    df: pd.DataFrame,
    column: str
) -> pd.DataFrame:
    df = df.copy()
    before = df[column].copy()

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.replace(
            r"\s+", " ", regex=True
        )
    )

    changed = int(
        (
            before.astype(str)
            != df[column].astype(str)
        ).sum()
    )

    add_cleaning_log(
        "Standardize text",
        f"Trimmed and normalized whitespace in "
        f"'{column}' ({changed} values changed)."
    )

    return df


def detect_outliers_iqr(
    series: pd.Series
) -> pd.Series:
    values = pd.to_numeric(
        series, errors="coerce"
    ).dropna()

    if len(values) < 4:
        return pd.Series(
            False, index=series.index
        )

    q1, q3 = (
        values.quantile(0.25),
        values.quantile(0.75)
    )

    iqr = q3 - q1
    lower, upper = (
        q1 - 1.5 * iqr, q3 + 1.5 * iqr
    )

    numeric = pd.to_numeric(
        series, errors="coerce"
    )

    return (
        (numeric < lower) | (numeric > upper)
    )


def detect_outliers_zscore(
    series: pd.Series,
    threshold: float = 3.0
) -> pd.Series:
    values = pd.to_numeric(
        series, errors="coerce"
    )

    mean, std = values.mean(), values.std()

    if pd.isna(std) or std == 0:
        return pd.Series(
            False, index=series.index
        )

    z = (values - mean) / std

    return z.abs() > threshold


# ---------------------------------------------------------------------------
# Chart safety helpers
# ---------------------------------------------------------------------------

def coerce_numeric_series(
    series: pd.Series
) -> pd.Series:
    return pd.to_numeric(
        series, errors="coerce"
    )


def safe_size_values(
    series: pd.Series,
    max_points: int = 60
) -> np.ndarray:
    """
    Converts any column into valid Plotly bubble sizes.
    Negative values are converted to absolute values.
    """

    values = coerce_numeric_series(series)

    if values.notna().sum() == 0:
        return np.full(len(series), 8.0)

    values = values.fillna(values.median())
    values = values.abs()

    lower = np.nanpercentile(values, 5)
    upper = np.nanpercentile(values, 95)

    if (
        not np.isfinite(lower)
        or not np.isfinite(upper)
        or upper <= lower
    ):
        return np.full(len(values), 10.0)

    scaled = 6 + (
        (values - lower) / (upper - lower)
    ) * 34

    scaled = np.clip(scaled, 4, 40)

    if len(scaled) > max_points:
        scaled = np.clip(scaled, 6, 22)

    return np.nan_to_num(
        scaled, nan=8.0
    )


def safe_color_values(
    series: pd.Series
) -> pd.Series:
    """
    Converts any column into a Plotly-safe
    color/group value.
    """

    if pd.api.types.is_datetime64_any_dtype(
        series
    ):
        return series.dt.strftime("%Y-%m-%d")

    if pd.api.types.is_bool_dtype(series):
        return series.astype(str)

    if pd.api.types.is_numeric_dtype(series):
        return series.astype(str)

    return series.astype(str)


def prepare_chart_data(
    df: pd.DataFrame,
    chart: str,
    x: Optional[str],
    y: Optional[str],
    color: Optional[str] = None,
    size: Optional[str] = None,
    aggregation: str = "Sum",
) -> pd.DataFrame:
    """
    Prepares a safe plotting DataFrame for every
    chart type.
    """

    data = df.copy()

    required = [
        c for c in [x, y, color, size] if c
    ]

    missing = [
        c for c in required
        if c not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Column(s) not found: "
            f"{', '.join(missing)}"
        )

    # Automatically detect and convert date-like
    # text columns.
    for column in [x, y, color]:
        if column and column in data.columns:
            if not pd.api.types.is_datetime64_any_dtype(
                data[column]
            ):
                parsed = pd.to_datetime(
                    data[column],
                    errors="coerce",
                    format="mixed"
                )

                parsed_ratio = (
                    parsed.notna().mean()
                    if len(parsed) else 0
                )

                if parsed_ratio > 0.75:
                    data[column] = parsed

    # Convert numeric text columns where needed.
    if y and chart in [
        "Vertical bar", "Horizontal bar",
        "Grouped bar", "Stacked bar",
        "100% stacked bar", "Line",
        "Multi-line", "Area", "Stacked area",
        "Moving average", "Histogram", "Box",
        "Violin", "Density", "ECDF", "Scatter",
        "Bubble", "Scatter matrix",
        "Correlation heatmap", "Pie", "Donut",
        "Treemap", "Funnel", "Ridgeline",
        "Q-Q plot", "Residual plot",
        "Anomaly plot", "Waterfall", "Pareto",
        "Lollipop"
    ]:
        if (
            y in data.columns
            and not pd.api.types.is_numeric_dtype(
                data[y]
            )
        ):
            data[y] = coerce_numeric_series(
                data[y]
            )

    if size and size in data.columns:
        data[size] = coerce_numeric_series(
            data[size]
        )

    # Aggregate categorical charts.
    aggregate_charts = [
        "Vertical bar", "Horizontal bar",
        "Grouped bar", "Stacked bar",
        "100% stacked bar", "Pie", "Donut",
        "Treemap", "Funnel", "Waterfall",
        "Pareto", "Lollipop"
    ]

    if (
        x and y
        and chart in aggregate_charts
        and aggregation != "None"
    ):
        agg_map = {
            "Sum": "sum",
            "Mean": "mean",
            "Median": "median",
            "Count": "count",
            "Minimum": "min",
            "Maximum": "max"
        }

        agg_func = agg_map.get(
            aggregation, "sum"
        )

        if agg_func == "count":
            data[y] = data[y].astype(str)

            grouped = data.groupby(
                x, dropna=False
            )[y].count().reset_index()

        else:
            grouped = data.groupby(
                x, dropna=False
            )[y].agg(agg_func).reset_index()

        grouped.columns = [x, y]
        data = grouped

    # Remove invalid plotting rows.
    plot_columns = [
        c for c in [x, y, color, size] if c
    ]

    for column in [x, y, size]:
        if column and column in data.columns:
            numeric = coerce_numeric_series(
                data[column]
            )

            if numeric.notna().any():
                data[column] = data[
                    column
                ].replace(
                    [np.inf, -np.inf], np.nan
                )

    if plot_columns:
        data = data.dropna(
            subset=plot_columns
        )

    if data.empty:
        raise ValueError(
            "No valid rows remain after cleaning "
            "the selected columns."
        )

    return data


# ---------------------------------------------------------------------------
# Visualization recommendation engine
# ---------------------------------------------------------------------------

CHART_CATALOG = [
    "Vertical bar", "Horizontal bar",
    "Grouped bar", "Stacked bar",
    "100% stacked bar", "Line", "Multi-line",
    "Area", "Stacked area", "Moving average",
    "Histogram", "Box", "Violin", "Density",
    "ECDF", "Scatter", "Bubble",
    "Scatter matrix", "Correlation heatmap",
    "Pie", "Donut", "Treemap", "Funnel",
    "Categorical heatmap", "Ridgeline",
    "Q-Q plot", "Residual plot",
    "Anomaly plot", "Gantt", "Map",
    "Waterfall", "Pareto", "Lollipop"
]


def recommend_visualizations(
    df: pd.DataFrame,
    types: Dict[str, str]
) -> List[Dict[str, Any]]:
    numeric = [
        c for c, t in types.items()
        if t == "numeric"
    ]

    categorical = [
        c for c, t in types.items()
        if t == "categorical"
        and unique_count(df[c]) <= 50
    ]

    dates = [
        c for c, t in types.items()
        if t == "datetime"
    ]

    recommendations = []

    def add(
        chart: str,
        x: Optional[str],
        y: Optional[str],
        score: int,
        reason: str,
        suitable: bool = True
    ):
        recommendations.append({
            "chart": chart,
            "x": x,
            "y": y,
            "score": score,
            "reason": reason,
            "suitable": suitable
        })

    if categorical and numeric:
        cat, num = (
            categorical[0], numeric[0]
        )

        add(
            "Horizontal bar", cat, num, 96,
            f"Compares {num} across categories "
            f"in {cat}."
        )

        add(
            "Vertical bar", cat, num, 92,
            f"Useful for comparing {num} by {cat}."
        )

        add(
            "Box", cat, num, 88,
            f"Shows the distribution and outliers "
            f"of {num} within each {cat}."
        )

        add(
            "Violin", cat, num, 82,
            f"Reveals distribution shape of {num} "
            f"across {cat}."
        )

        add(
            "Treemap", cat, num, 78,
            f"Shows proportional contribution of "
            f"each {cat} to total {num}."
        )

    if dates and numeric:
        date, num = dates[0], numeric[0]

        add(
            "Line", date, num, 97,
            f"Displays how {num} changes over "
            f"{date}."
        )

        add(
            "Area", date, num, 88,
            f"Emphasizes cumulative magnitude of "
            f"{num} over time."
        )

        add(
            "Moving average", date, num, 85,
            f"Smooths short-term variation in "
            f"{num}."
        )

    if len(numeric) >= 2:
        add(
            "Correlation heatmap", None, None, 94,
            "Identifies linear relationships among "
            "numeric variables."
        )

        add(
            "Scatter matrix", None, None, 85,
            "Explores pairwise relationships and "
            "distributions."
        )

        add(
            "Scatter", numeric[0], numeric[1], 90,
            f"Examines the relationship between "
            f"{numeric[0]} and {numeric[1]}."
        )

        add(
            "Bubble", numeric[0], numeric[1], 72,
            "Adds a third numeric dimension through "
            "bubble size."
        )

    if numeric:
        add(
            "Histogram", numeric[0], None, 95,
            f"Shows the distribution of "
            f"{numeric[0]}."
        )

        add(
            "Box", None, numeric[0], 90,
            f"Summarizes spread, median, and outliers "
            f"for {numeric[0]}."
        )

    if len(categorical) >= 2:
        add(
            "Categorical heatmap",
            categorical[0], categorical[1], 78,
            "Reveals frequency patterns between "
            "categorical fields."
        )

    return sorted(
        recommendations,
        key=lambda item: item["score"],
        reverse=True
    )


def chart_is_suitable(
    chart: str,
    df: pd.DataFrame,
    types: Dict[str, str],
    x: Optional[str],
    y: Optional[str]
) -> Tuple[bool, str]:
    """
    Provides guidance but does not permanently
    block charts.
    """

    numeric = [
        c for c, t in types.items()
        if t == "numeric"
    ]

    categorical = [
        c for c, t in types.items()
        if t == "categorical"
    ]

    dates = [
        c for c, t in types.items()
        if t == "datetime"
    ]

    if chart in [
        "Line", "Area", "Moving average",
        "Multi-line", "Stacked area"
    ]:
        if not dates:
            return False, (
                "Tip: Select a date/time column as X."
            )

        if not numeric:
            return False, (
                "Tip: Select a numeric column as Y."
            )

    if chart in [
        "Histogram", "Box", "Violin", "Density",
        "ECDF", "Q-Q plot"
    ]:
        if not numeric:
            return False, (
                "Tip: Select a numeric column."
            )

    if chart in [
        "Scatter", "Bubble", "Residual plot"
    ]:
        if len(numeric) < 2:
            return False, (
                "Tip: Scatter charts work best with "
                "two numeric columns."
            )

    if chart in [
        "Pie", "Donut", "Treemap", "Funnel"
    ]:
        if not categorical or not numeric:
            return False, (
                "Tip: Select a category and numeric "
                "column."
            )

    if (
        chart == "Correlation heatmap"
        and len(numeric) < 2
    ):
        return False, (
            "Tip: Correlation requires at least two "
            "numeric columns."
        )

    if (
        chart == "Categorical heatmap"
        and len(categorical) < 2
    ):
        return False, (
            "Tip: Select two categorical columns."
        )

    if chart == "Gantt" and len(dates) < 2:
        return False, (
            "Tip: Gantt charts require two date "
            "columns."
        )

    if chart == "Map":
        geo = [
            c for c in df.columns
            if any(
                word in c.lower()
                for word in ["lat", "lon"]
            )
        ]

        if len(geo) < 2:
            return False, (
                "Tip: Maps require latitude and "
                "longitude columns."
            )

    return True, "Suitable."


# ---------------------------------------------------------------------------
# Universal chart builder
# ---------------------------------------------------------------------------

def build_chart(
    df: pd.DataFrame,
    chart: str,
    x: Optional[str],
    y: Optional[str],
    color: Optional[str] = None,
    size: Optional[str] = None,
    aggregation: str = "Sum",
    title: Optional[str] = None,
    orientation: str = "Vertical",
) -> go.Figure:
    """
    Universal, type-aware chart builder.
    """

    data = prepare_chart_data(
        df=df,
        chart=chart,
        x=x,
        y=y,
        color=color,
        size=size,
        aggregation=aggregation,
    )

    if color and color in data.columns:
        data["_color_label"] = safe_color_values(
            data[color]
        )

        color = "_color_label"

    size_values = None

    if size and size in data.columns:
        size_values = safe_size_values(
            data[size]
        )

    fig = None

    if chart == "Vertical bar":
        fig = px.bar(
            data, x=x, y=y, color=color,
            title=(
                title
                or f"{y or 'Value'} by "
                f"{x or 'Category'}"
            )
        )

    elif chart == "Horizontal bar":
        fig = px.bar(
            data, x=y, y=x, color=color,
            orientation="h",
            title=(
                title
                or f"{y or 'Value'} by "
                f"{x or 'Category'}"
            )
        )

    elif chart == "Grouped bar":
        fig = px.bar(
            data, x=x, y=y, color=color,
            barmode="group",
            title=(
                title
                or f"{y or 'Value'} by "
                f"{x or 'Category'}"
            )
        )

    elif chart == "Stacked bar":
        fig = px.bar(
            data, x=x, y=y, color=color,
            barmode="stack",
            title=(
                title
                or f"Stacked {y or 'Value'} by "
                f"{x or 'Category'}"
            )
        )

    elif chart == "100% stacked bar":
        fig = px.bar(
            data, x=x, y=y, color=color,
            barmode="relative",
            title=(
                title
                or f"100% Stacked {y or 'Value'} by "
                f"{x or 'Category'}"
            )
        )

    elif chart in [
        "Line", "Multi-line", "Area",
        "Stacked area", "Moving average"
    ]:
        if x and pd.api.types.is_datetime64_any_dtype(
            data[x]
        ):
            data = data.sort_values(x)

        if chart == "Line":
            fig = px.line(
                data, x=x, y=y, color=color,
                markers=True
            )

        elif chart == "Multi-line":
            fig = px.line(
                data, x=x, y=y, color=color,
                markers=True
            )

        elif chart == "Area":
            fig = px.area(
                data, x=x, y=y, color=color
            )

        elif chart == "Stacked area":
            fig = px.area(
                data, x=x, y=y, color=color
            )

        elif chart == "Moving average":
            window = int(
                st.session_state.analysis_context.get(
                    "moving_average_window", 7
                )
            )

            data[f"{y}_moving_average"] = (
                data[y]
                .rolling(
                    window=min(
                        window, max(1, len(data))
                    ),
                    min_periods=1
                )
                .mean()
            )

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=data[x], y=data[y],
                mode="lines+markers",
                name=y, opacity=0.45
            ))

            fig.add_trace(go.Scatter(
                x=data[x],
                y=data[f"{y}_moving_average"],
                mode="lines",
                name=(
                    f"{window}-period moving average"
                )
            ))

    elif chart == "Histogram":
        fig = px.histogram(
            data, x=y or x, color=color, nbins=40,
            title=(
                title
                or f"Distribution of {y or x}"
            )
        )

    elif chart == "Box":
        fig = px.box(
            data, x=x, y=y, color=color,
            title=(
                title
                or f"Box Plot: {y or 'Value'}"
            )
        )

    elif chart == "Violin":
        fig = px.violin(
            data, x=x, y=y, color=color, box=True,
            title=(
                title
                or f"Violin Plot: {y or 'Value'}"
            )
        )

    elif chart == "Density":
        fig = px.density_contour(
            data, x=y or x, color=color,
            title=(
                title or f"Density: {y or x}"
            )
        )

    elif chart == "ECDF":
        fig = px.ecdf(
            data, x=y or x, color=color,
            title=(
                title or f"ECDF: {y or x}"
            )
        )

    elif chart in ["Scatter", "Bubble"]:
        fig = px.scatter(
            data, x=x, y=y, color=color,
            size=(
                size_values
                if size_values is not None
                else None
            ),
            hover_data=[
                c for c in data.columns
                if c not in [x, y]
            ][:5],
            title=(
                title
                or f"{y or 'Value'} vs "
                f"{x or 'Index'}"
            )
        )

    elif chart == "Scatter matrix":
        numeric_columns = [
            c for c in data.columns
            if pd.api.types.is_numeric_dtype(
                data[c]
            )
        ][:6]

        if len(numeric_columns) < 2:
            raise ValueError(
                "Scatter matrix requires at least two "
                "numeric columns."
            )

        fig = px.scatter_matrix(
            data, dimensions=numeric_columns,
            color=color,
            title=title or "Scatter Matrix"
        )

    elif chart == "Correlation heatmap":
        numeric_data = data.select_dtypes(
            include=[np.number]
        )

        if numeric_data.shape[1] < 2:
            raise ValueError(
                "Correlation heatmap requires at least "
                "two numeric columns."
            )

        correlation = numeric_data.corr(
            numeric_only=True
        )

        fig = px.imshow(
            correlation, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title=(
                title or "Correlation Heatmap"
            )
        )

    elif chart in ["Pie", "Donut"]:
        fig = px.pie(
            data, names=x, values=y,
            hole=(
                0.45 if chart == "Donut" else 0
            ),
            title=(
                title
                or f"{y or 'Value'} by "
                f"{x or 'Category'}"
            )
        )

    elif chart == "Treemap":
        fig = px.treemap(
            data, path=[x], values=y, color=y,
            title=(
                title
                or f"Treemap: {y or 'Value'} by "
                f"{x or 'Category'}"
            )
        )

    elif chart == "Funnel":
        fig = px.funnel(
            data, x=y, y=x,
            title=(
                title
                or f"Funnel: {y or 'Value'} by "
                f"{x or 'Stage'}"
            )
        )

    elif chart == "Categorical heatmap":
        if not x or not y:
            raise ValueError(
                "Categorical heatmap requires two "
                "categorical columns."
            )

        cross_tab = pd.crosstab(
            data[x], data[y]
        )

        fig = px.imshow(
            cross_tab, text_auto=True,
            color_continuous_scale="Blues",
            title=title or f"{x} vs {y}"
        )

    elif chart == "Ridgeline":
        fig = px.histogram(
            data, x=y, color=x, marginal="rug",
            title=(
                title
                or f"Ridgeline-style Distribution: {y}"
            )
        )

    elif chart == "Q-Q plot":
        values = coerce_numeric_series(
            data[y]
        ).dropna()

        if len(values) < 3:
            raise ValueError(
                "Q-Q plot requires at least three "
                "numeric values."
            )

        if not SCIPY_AVAILABLE:
            raise ValueError(
                "Q-Q plot requires SciPy. Install it "
                "with: pip install scipy"
            )

        theoretical, ordered = stats.probplot(
            values, dist="norm"
        )

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=theoretical[0],
            y=theoretical[1],
            mode="markers",
            name="Sample quantiles"
        ))

        slope = (
            theoretical[1][1]
            if len(theoretical[1]) > 1 else 1
        )

        intercept = (
            theoretical[1][0]
            if len(theoretical[1]) else 0
        )

        fig.add_trace(go.Scatter(
            x=theoretical[0],
            y=(
                slope
                * np.array(theoretical[0])
                + intercept
            ),
            mode="lines",
            name="Reference line"
        ))

    elif chart == "Residual plot":
        numeric_data = data[[x, y]].dropna()

        if len(numeric_data) < 3:
            raise ValueError(
                "Residual plot requires at least three "
                "valid observations."
            )

        slope, intercept = np.polyfit(
            numeric_data[x], numeric_data[y], 1
        )

        residuals = numeric_data[y] - (
            slope * numeric_data[x] + intercept
        )

        fig = px.scatter(
            x=numeric_data[x], y=residuals,
            title=(
                title
                or f"Residuals: {y} vs {x}"
            )
        )

        fig.add_hline(
            y=0, line_dash="dash"
        )

    elif chart == "Anomaly plot":
        if not y:
            raise ValueError(
                "Anomaly plot requires a numeric Y "
                "column."
            )

        anomaly_mask = detect_outliers_iqr(
            data[y]
        )

        data["_anomaly"] = (
            anomaly_mask.astype(str)
        )

        fig = px.scatter(
            data, x=x, y=y, color="_anomaly",
            color_discrete_map={
                "True": "red",
                "False": "#1f77b4"
            },
            title=(
                title or f"Anomaly Plot: {y}"
            )
        )

    elif chart == "Gantt":
        if not x or not y:
            raise ValueError(
                "Gantt chart requires start and end "
                "date columns."
            )

        data[x] = pd.to_datetime(
            data[x], errors="coerce"
        )

        data[y] = pd.to_datetime(
            data[y], errors="coerce"
        )

        data = data.dropna(
            subset=[x, y]
        )

        if data.empty:
            raise ValueError(
                "No valid start and end dates found "
                "for the Gantt chart."
            )

        data["_task"] = data.index.astype(str)

        fig = px.timeline(
            data, x_start=x, x_end=y, y="_task",
            color=color,
            title=title or "Gantt Timeline"
        )

    elif chart == "Map":
        latitude_column = next(
            (
                c for c in data.columns
                if "lat" in c.lower()
            ),
            None
        )

        longitude_column = next(
            (
                c for c in data.columns
                if "lon" in c.lower()
            ),
            None
        )

        if (
            not latitude_column
            or not longitude_column
        ):
            raise ValueError(
                "Map requires latitude and longitude "
                "columns."
            )

        data[latitude_column] = coerce_numeric_series(
            data[latitude_column]
        )

        data[longitude_column] = coerce_numeric_series(
            data[longitude_column]
        )

        data = data.dropna(
            subset=[
                latitude_column,
                longitude_column
            ]
        )

        fig = px.scatter_map(
            data,
            lat=latitude_column,
            lon=longitude_column,
            color=color,
            size=(
                size_values
                if size_values is not None
                else None
            ),
            hover_data=[
                c for c in data.columns
                if c not in [
                    latitude_column,
                    longitude_column
                ]
            ][:5],
            title=(
                title or "Geographic Scatter Map"
            ),
            zoom=1
        )

    elif chart == "Waterfall":
        if not x or not y:
            raise ValueError(
                "Waterfall chart requires a category "
                "and numeric column."
            )

        values = data.groupby(x)[y].sum().sort_values(
            ascending=False
        )

        fig = go.Figure(
            go.Waterfall(
                x=values.index.astype(str),
                y=values.values,
                connector={
                    "line": {
                        "color": "#64748b"
                    }
                }
            )
        )

    elif chart == "Pareto":
        if not x or not y:
            raise ValueError(
                "Pareto chart requires a category and "
                "numeric column."
            )

        grouped = data.groupby(x)[y].sum().sort_values(
            ascending=False
        )

        cumulative = (
            grouped.cumsum()
            / grouped.sum()
            * 100
        )

        fig = go.Figure()

        fig.add_bar(
            x=grouped.index.astype(str),
            y=grouped.values,
            name=y
        )

        fig.add_scatter(
            x=grouped.index.astype(str),
            y=cumulative.values,
            name="Cumulative %",
            yaxis="y2",
            mode="lines+markers"
        )

        fig.update_layout(
            yaxis2={
                "overlaying": "y",
                "side": "right",
                "range": [0, 105]
            }
        )

    elif chart == "Lollipop":
        if not x or not y:
            raise ValueError(
                "Lollipop chart requires a category and "
                "numeric column."
            )

        grouped = data.groupby(x)[y].sum().sort_values()

        fig = go.Figure()

        fig.add_scatter(
            x=grouped.values,
            y=grouped.index.astype(str),
            mode="markers",
            name=y
        )

        for index, value in enumerate(
            grouped.values
        ):
            fig.add_shape(
                type="line",
                x0=0,
                x1=value,
                y0=grouped.index[index],
                y1=grouped.index[index],
                line={
                    "color": "#94a3b8"
                }
            )

    else:
        raise ValueError(
            f"Chart type '{chart}' is not implemented."
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(
            l=20, r=20, t=55, b=35
        ),
        hovermode="closest",
        font={
            "family": "Inter, Arial, sans-serif"
        },
        title=title or chart
    )

    return fig


# ---------------------------------------------------------------------------
# Analytical engine
# ---------------------------------------------------------------------------

def apply_filters(
    df: pd.DataFrame,
    filters: List[Dict[str, Any]]
) -> pd.DataFrame:
    filtered = df.copy()

    for flt in filters or []:
        column = flt.get("column")
        operator = flt.get("operator")
        value = flt.get("value")

        if column not in filtered.columns:
            continue

        if operator == "equals":
            filtered = filtered[
                filtered[column]
                .astype(str)
                .str.lower()
                == str(value).lower()
            ]

        elif operator == "not_equals":
            filtered = filtered[
                filtered[column]
                .astype(str)
                .str.lower()
                != str(value).lower()
            ]

        elif operator == "greater_than":
            filtered = filtered[
                pd.to_numeric(
                    filtered[column],
                    errors="coerce"
                ) > safe_float(value)
            ]

        elif operator == "less_than":
            filtered = filtered[
                pd.to_numeric(
                    filtered[column],
                    errors="coerce"
                ) < safe_float(value)
            ]

        elif operator == "contains":
            filtered = filtered[
                filtered[column]
                .astype(str)
                .str.contains(
                    str(value),
                    case=False,
                    na=False
                )
            ]

    return filtered


def execute_analytical_plan(
    df: pd.DataFrame,
    plan: Dict[str, Any]
) -> Dict[str, Any]:
    operation = plan.get(
        "operation", "summary"
    )

    column = plan.get("column")
    group_by = plan.get("group_by")
    filters = plan.get("filters", [])
    chart = plan.get("chart")

    data = apply_filters(df, filters)

    if data.empty:
        return {
            "error": (
                "No rows remain after applying the "
                "requested filters."
            ),
            "plan": plan
        }

    result = {
        "operation": operation,
        "filters": filters,
        "rows_analyzed": int(len(data))
    }

    if operation in [
        "sum", "mean", "median", "count",
        "min", "max"
    ] and column:
        if column not in data.columns:
            return {
                "error": (
                    f"Column '{column}' was not found."
                ),
                "plan": plan
            }

        series = (
            pd.to_numeric(
                data[column], errors="coerce"
            ).dropna()
            if operation != "count"
            else data[column].dropna()
        )

        if series.empty:
            return {
                "error": (
                    f"No valid values found in "
                    f"'{column}'."
                ),
                "plan": plan
            }

        value = {
            "sum": series.sum(),
            "mean": series.mean(),
            "median": series.median(),
            "count": len(series),
            "min": series.min(),
            "max": series.max()
        }[operation]

        result.update({
            "value": make_json_safe(value),
            "column": column,
            "group_by": group_by,
            "chart": chart
        })

        if group_by and group_by in data.columns:
            grouped = data.groupby(
                group_by, dropna=False
            )[column].agg(
                operation
                if operation != "count"
                else "count"
            ).reset_index()

            grouped.columns = [
                group_by, "value"
            ]

            grouped = grouped.sort_values(
                "value", ascending=False
            )

            result.update({
                "table": grouped,
                "top_item": make_json_safe(
                    grouped.iloc[0].to_dict()
                ) if len(grouped) else None,
                "chart": (
                    chart or "Horizontal bar"
                ),
                "chart_x": group_by,
                "chart_y": "value"
            })

        return result

    if operation == "trend" and column and group_by:
        if (
            column not in data.columns
            or group_by not in data.columns
        ):
            return {
                "error": (
                    "Trend analysis requires valid date "
                    "and numeric columns."
                ),
                "plan": plan
            }

        data = data.copy()

        data[group_by] = pd.to_datetime(
            data[group_by], errors="coerce"
        )

        data = data.dropna(
            subset=[group_by, column]
        )

        data["period"] = data[
            group_by
        ].dt.to_period("M").astype(str)

        trend = data.groupby(
            "period"
        )[column].sum().reset_index()

        result.update({
            "table": trend,
            "chart": chart or "Line",
            "chart_x": "period",
            "chart_y": column,
            "column": column,
            "group_by": group_by
        })

        return result

    if operation == "correlation":
        numeric_df = data.select_dtypes(
            include=[np.number]
        )

        if numeric_df.shape[1] < 2:
            return {
                "error": (
                    "Correlation requires at least two "
                    "numeric columns."
                ),
                "plan": plan
            }

        corr = numeric_df.corr()

        result.update({
            "correlation_matrix": corr.round(
                3
            ).to_dict(),
            "chart": "Correlation heatmap"
        })

        return result

    if operation == "distribution" and column:
        if column not in data.columns:
            return {
                "error": (
                    f"Column '{column}' was not found."
                ),
                "plan": plan
            }

        series = pd.to_numeric(
            data[column], errors="coerce"
        ).dropna()

        if series.empty:
            return {
                "error": (
                    f"No numeric values available in "
                    f"'{column}'."
                ),
                "plan": plan
            }

        result.update({
            "mean": safe_float(series.mean()),
            "median": safe_float(series.median()),
            "std": safe_float(series.std()),
            "min": safe_float(series.min()),
            "max": safe_float(series.max()),
            "chart": chart or "Histogram",
            "chart_y": column,
            "column": column
        })

        return result

    if operation == "top_n" and column and group_by:
        if (
            column not in data.columns
            or group_by not in data.columns
        ):
            return {
                "error": (
                    "Top-N analysis requires valid metric "
                    "and category columns."
                ),
                "plan": plan
            }

        grouped = data.groupby(
            group_by, dropna=False
        )[column].sum().reset_index()

        grouped.columns = [
            group_by, "value"
        ]

        grouped = grouped.sort_values(
            "value", ascending=False
        )

        n = int(plan.get("limit", 10))

        result.update({
            "table": grouped.head(n),
            "chart": chart or "Horizontal bar",
            "chart_x": group_by,
            "chart_y": "value",
            "column": column,
            "group_by": group_by,
            "top_item": make_json_safe(
                grouped.iloc[0].to_dict()
            ) if len(grouped) else None
        })

        return result

    if operation == "summary":
        numeric_summary = data.describe(
            include=[np.number]
        ).T.reset_index()

        numeric_summary.rename(
            columns={"index": "column"},
            inplace=True
        )

        result.update({
            "table": numeric_summary,
            "chart": None
        })

        return result

    return {
        "error": (
            f"Unsupported analytical operation: "
            f"{operation}."
        ),
        "plan": plan
    }


# ---------------------------------------------------------------------------
# Groq AI functions
# ---------------------------------------------------------------------------

def get_groq_client() -> Optional[Groq]:
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        api_key = st.secrets.get("GROQ_API_KEY", None)

    if not api_key:
        return None

    if not GROQ_AVAILABLE:
        return None

    try:
        return Groq(api_key=api_key)

    except Exception:
        return None

def groq_chat(
    client: Groq,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_tokens: int = 800
) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return response.choices[0].message.content.strip()

def get_available_groq_models(
    client: Optional[Groq]
) -> List[str]:
    """
    Returns model IDs available to the current Groq API key.
    """

    default_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it"
    ]

    if not client:
        return default_models

    try:
        response = client.models.list()

        models = [
            model.id
            for model in response.data
            if getattr(model, "active", True)
        ]

        if models:
            return sorted(models)

        return default_models

    except Exception:
        return default_models


def rule_based_plan(
    question: str,
    df: pd.DataFrame,
    types: Dict[str, str],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Reliable rule-based planner.
    """

    q = question.lower().strip()

    numeric = [
        c for c, t in types.items()
        if t == "numeric"
    ]

    categorical = [
        c for c, t in types.items()
        if t == "categorical"
        and unique_count(df[c]) <= 50
    ]

    dates = [
        c for c, t in types.items()
        if t == "datetime"
    ]

    plan = {
        "operation": "summary",
        "filters": [],
        "limit": 10
    }

    mentioned_column = next(
        (
            c for c in df.columns
            if c.lower() in q
            or c.lower().replace("_", " ") in q
        ),
        None
    )

    mentioned_category = next(
        (
            c for c in categorical
            if c.lower() in q
            or c.lower().replace("_", " ") in q
        ),
        None
    )

    mentioned_date = next(
        (
            c for c in dates
            if c.lower() in q
            or c.lower().replace("_", " ") in q
        ),
        None
    )

    if any(
        word in q for word in [
            "correlation", "relationship",
            "related", "association", "correlate"
        ]
    ):
        plan.update(
            {"operation": "correlation"}
        )

        return plan

    if any(
        word in q for word in [
            "trend", "over time", "monthly",
            "yearly", "growth", "time series",
            "by date", "per month"
        ]
    ):
        date_col = mentioned_date or (
            dates[0] if dates else None
        )

        metric = mentioned_column or (
            numeric[0] if numeric else None
        )

        if date_col and metric:
            plan.update({
                "operation": "trend",
                "column": metric,
                "group_by": date_col,
                "chart": "Line"
            })

            return plan

    if any(
        word in q for word in [
            "top", "highest", "best", "largest",
            "most", "lowest", "worst", "smallest",
            "least", "maximum", "minimum"
        ]
    ):
        metric = mentioned_column or (
            numeric[0] if numeric else None
        )

        category = mentioned_category or (
            categorical[0] if categorical else None
        )

        if metric and category:
            plan.update({
                "operation": "top_n",
                "column": metric,
                "group_by": category,
                "chart": "Horizontal bar",
                "limit": 10
            })

            return plan

    if any(
        word in q for word in [
            "average", "mean", "typical", "avg"
        ]
    ):
        metric = mentioned_column or (
            numeric[0] if numeric else None
        )

        if metric:
            plan.update({
                "operation": "mean",
                "column": metric,
                "group_by": mentioned_category,
                "chart": "Vertical bar"
            })

            return plan

    if any(
        word in q for word in [
            "total", "sum", "revenue", "sales",
            "profit", "quantity", "amount",
            "price", "cost"
        ]
    ):
        metric = mentioned_column or (
            numeric[0] if numeric else None
        )

        if metric:
            plan.update({
                "operation": "sum",
                "column": metric,
                "group_by": mentioned_category,
                "chart": "Vertical bar"
            })

            return plan

    if any(
        word in q for word in [
            "distribution", "spread", "outlier",
            "anomaly", "range", "variation"
        ]
    ):
        metric = mentioned_column or (
            numeric[0] if numeric else None
        )

        if metric:
            plan.update({
                "operation": "distribution",
                "column": metric,
                "chart": "Histogram"
            })

            return plan

    if any(
        word in q for word in [
            "chart", "plot", "visualize", "graph",
            "heatmap", "scatter", "bar"
        ]
    ):
        if "scatter" in q and len(numeric) >= 2:
            plan.update({
                "operation": "summary",
                "chart": "Scatter",
                "column": numeric[0],
                "group_by": (
                    categorical[0]
                    if categorical else None
                )
            })

        elif "heatmap" in q:
            plan.update({
                "operation": "correlation",
                "chart": "Correlation heatmap"
            })

        elif categorical and numeric:
            plan.update({
                "operation": "top_n",
                "column": numeric[0],
                "group_by": categorical[0],
                "chart": "Horizontal bar"
            })

        return plan

    if any(
        word in q for word in [
            "its", "that", "same", "previous",
            "compare it"
        ]
    ) and context:
        prior_metric = context.get(
            "last_metric"
        )

        prior_category = context.get(
            "last_category"
        )

        if dates and prior_metric:
            plan.update({
                "operation": "trend",
                "column": prior_metric,
                "group_by": dates[0],
                "chart": "Line"
            })

            if (
                prior_category
                and context.get("last_top_value")
            ):
                plan["filters"] = [{
                    "column": prior_category,
                    "operator": "equals",
                    "value": context.get(
                        "last_top_value"
                    )
                }]

            return plan

    return plan


def extract_plan_with_groq(
    question: str,
    df: pd.DataFrame,
    types: Dict[str, str],
    client: Groq,
    model: str
) -> Optional[Dict[str, Any]]:
    """
    Uses Groq only to choose a safe analysis plan.
    The actual calculation is always performed by
    Pandas.
    """

    columns = []

    for column in df.columns:
        sample = (
            df[column]
            .dropna()
            .head(5)
            .astype(str)
            .tolist()
        )

        columns.append({
            "name": column,
            "type": types.get(
                column, "categorical"
            ),
            "unique_values": unique_count(
                df[column]
            ),
            "sample_values": sample
        })

    schema = json.dumps(
        columns[:40], indent=2
    )

    prompt = f"""You are a strict data-analysis planner.

Dataset columns:
{schema}

User question:
{question}

Choose exactly one operation from:
sum, mean, median, count, min, max, trend, correlation,
distribution, top_n, summary

Rules:
- Use only column names that exist exactly in the dataset.
- If the user asks for total, revenue, sales, profit, quantity,
  price, or amount, use "sum".
- If the user asks for average, mean, or typical value, use "mean".
- If the user asks for highest, lowest, top, best, worst, largest,
  or smallest, use "top_n".
- If the user asks about trend, growth, monthly performance, or
  over time, use "trend".
- If the user asks about correlation or relationship, use
  "correlation".
- If the user asks about distribution, spread, outliers, or
  anomalies, use "distribution".
- If the question is unclear, use "summary".
- Do not create executable code.
- Return ONLY valid JSON. No explanation, no markdown.

JSON format:
{{
  "operation": "one allowed operation",
  "column": "exact column name or null",
  "group_by": "exact column name or null",
  "filters": [],
  "chart": "Horizontal bar|Line|Histogram|Scatter|Correlation heatmap|null",
  "limit": 10
}}"""

    try:
        content = groq_chat(
            client=client,
            model=model,
            system_prompt=(
                "You are a precise data-analysis planner. "
                "Return only valid JSON."
            ),
            user_prompt=prompt,
            temperature=0,
            max_tokens=400
        )

        content = re.sub(
            r"^```json|```$",
            "",
            content,
            flags=re.MULTILINE
        ).strip()

        match = re.search(
            r"\{.*\}", content, re.DOTALL
        )

        if match:
            content = match.group(0)

        plan = json.loads(content)

        if not isinstance(plan, dict):
            return None

        allowed_operations = {
            "sum", "mean", "median", "count",
            "min", "max", "trend", "correlation",
            "distribution", "top_n", "summary"
        }

        operation = plan.get("operation")

        if operation not in allowed_operations:
            return None

        if plan.get("column") not in df.columns:
            plan["column"] = None

        if plan.get("group_by") not in df.columns:
            plan["group_by"] = None

        if operation in [
            "sum", "mean", "median", "min", "max",
            "distribution", "top_n"
        ] and not plan.get("column"):
            return None

        if (
            operation in ["top_n", "trend"]
            and not plan.get("group_by")
        ):
            return None

        return plan

    except Exception:
        return None


def classify_intent(
    question: str,
    has_data: bool
) -> str:
    q = question.lower()

    analytical_words = [
        "total", "average", "sum", "top", "highest",
        "lowest", "trend", "correlation",
        "distribution", "count", "by", "sales",
        "revenue", "profit"
    ]

    visualization_words = [
        "chart", "plot", "graph", "visualize",
        "heatmap", "scatter", "bar"
    ]

    followup_words = [
        "its", "that", "same", "previous",
        "compare it"
    ]

    if any(
        word in q
        for word in visualization_words
    ) and has_data:
        return "dataset_visualization"

    if any(
        word in q
        for word in analytical_words
    ) and has_data:
        return "dataset_analytical"

    if any(
        word in q for word in followup_words
    ) and st.session_state.analysis_context:
        return "follow_up"

    return "general_knowledge"


def generate_ai_insight(
    question: str,
    result: Dict[str, Any],
    client: Optional[Groq],
    model: str
) -> str:
    evidence = json.dumps(
        make_json_safe({
            k: v for k, v in result.items()
            if k != "table"
        }),
        default=str
    )[:4000]

    prompt = f"""Explain this verified data-analysis result clearly for a business user.

Question:
{question}

Verified result:
{evidence}

Rules:
- Use only the provided result.
- Do not claim causation from correlation.
- Be concise and practical.
- Do not invent values.

Format:
Finding:
Evidence:
Interpretation:
Recommendation:
Limitations:"""

    if client:
        try:
            return groq_chat(
                client=client,
                model=model,
                system_prompt=(
                    "You are a precise data-analysis "
                    "interpreter. Explain verified results "
                    "clearly without inventing facts."
                ),
                user_prompt=prompt,
                temperature=0.2,
                max_tokens=700
            )

        except Exception as exc:
            return (
                f"Groq explanation is unavailable ({exc}). "
                "The numerical result above was calculated "
                "directly from your dataset."
            )

    return (
        "Groq is not configured. The numerical result "
        "above was calculated directly from your dataset "
        "using Python and Pandas."
    )


def answer_general_question(
    question: str,
    client: Optional[Groq],
    model: str,
    df: Optional[pd.DataFrame] = None,
    has_data: bool = False
) -> str:
    """
    Answers both dataset and general questions.
    """

    dataset_context = ""

    if has_data and df is not None:
        types = infer_column_types(df)

        numeric = [
            c for c, t in types.items()
            if t == "numeric"
        ]

        categorical = [
            c for c, t in types.items()
            if t == "categorical"
        ]

        dates = [
            c for c, t in types.items()
            if t == "datetime"
        ]

        sample_rows = df.head(
            3
        ).to_dict(orient="records")

        dataset_context = f"""
Dataset context:
- Total rows: {len(df):,}
- Total columns: {len(df.columns)}
- Numeric columns: {', '.join(numeric[:20]) if numeric else 'None'}
- Categorical columns: {', '.join(categorical[:20]) if categorical else 'None'}
- Date columns: {', '.join(dates[:20]) if dates else 'None'}

Column samples:
{json.dumps(make_json_safe(sample_rows), default=str, indent=2)}

Important:
- If the user asks about this dataset, use the available columns
  and explain what analysis is possible.
- If the user asks a general question, answer it normally.
- Do not say the question is unrelated unless it truly cannot be answered.
- Do not invent dataset values.
"""

    if client:
        try:
            return groq_chat(
                client=client,
                model=model,
                system_prompt="""You are an intelligent, helpful data-analysis and general-purpose assistant.

Rules:
- Always answer the user directly.
- If a dataset is available, use its column names and samples when relevant.
- If the user asks about the dataset, suggest or explain the most relevant analysis.
- If the user asks a general question, answer normally with clear reasoning.
- Be concise, practical, and well structured.
- Clearly distinguish facts, assumptions, estimates, and uncertainty.
- Never invent dataset values or claim a calculation was performed unless calculation results are provided.
- If a dataset calculation is needed, explain which columns should be used and what analysis would answer the question.""",
                user_prompt=(
                    f"{dataset_context}\n\n"
                    f"User question:\n{question}"
                ),
                temperature=0.3,
                max_tokens=1000
            )

        except Exception as exc:
            return (
                f"**Groq AI request failed.**\n\n"
                f"Error: {exc}\n\n"
                "Check that your `GROQ_API_KEY` is correctly added "
                "in Streamlit Secrets and that the selected model "
                "name is available to your Groq account."
            )

    return (
        "**Groq is not configured.**\n\n"
        "Add your API key in Streamlit Secrets:\n\n"
        "```toml\nGROQ_API_KEY = \"your_groq_api_key_here\"\n```\n\n"
        "Dataset analysis, charts, cleaning, KPIs, anomaly detection, "
        "and reports still work without Groq."
    )


# ---------------------------------------------------------------------------
# KPI discovery
# ---------------------------------------------------------------------------

def discover_kpis(
    df: pd.DataFrame,
    types: Dict[str, str]
) -> List[Dict[str, Any]]:
    columns = {
        c.lower(): c for c in df.columns
    }

    dates = [
        c for c, t in types.items()
        if t == "datetime"
    ]

    kpis = []

    def find_column(
        candidates: List[str]
    ) -> Optional[str]:
        for candidate in candidates:
            if candidate in columns:
                return columns[candidate]

        return None

    revenue = find_column([
        "revenue", "sales", "sales amount",
        "amount", "total", "total revenue",
        "gross revenue"
    ])

    profit = find_column([
        "profit", "gross profit", "net profit",
        "margin"
    ])

    quantity = find_column([
        "quantity", "units", "units sold",
        "qty", "volume"
    ])

    customer = find_column([
        "customer", "customer id", "client",
        "user id", "user"
    ])

    order = find_column([
        "order", "order id", "transaction id",
        "invoice", "invoice id"
    ])

    cost = find_column([
        "cost", "cogs", "cost of goods sold",
        "expense", "expenses"
    ])

    date = dates[0] if dates else None

    if revenue:
        kpis.append({
            "name": f"Total {revenue}",
            "value": safe_float(
                df[revenue].sum()
            ),
            "formula": (
                "Sum of numeric values"
            ),
            "assumption": (
                f"'{revenue}' is treated as an additive "
                "business measure."
            )
        })

        kpis.append({
            "name": f"Average {revenue}",
            "value": safe_float(
                df[revenue].mean()
            ),
            "formula": (
                "Mean of non-missing values"
            ),
            "assumption": (
                "Each row represents a comparable "
                "transaction or record."
            )
        })

    if revenue and quantity:
        total_quantity = safe_float(
            df[quantity].sum()
        )

        if total_quantity:
            kpis.append({
                "name": "Average value per unit",
                "value": safe_float(
                    df[revenue].sum()
                    / total_quantity
                ),
                "formula": (
                    "Total revenue / total quantity"
                ),
                "assumption": (
                    "Revenue and quantity are aligned "
                    "at row level."
                )
            })

    if revenue and cost:
        total_revenue = safe_float(
            df[revenue].sum()
        )

        total_cost = safe_float(
            df[cost].sum()
        )

        if total_revenue:
            kpis.append({
                "name": "Gross margin",
                "value": round(
                    (
                        total_revenue - total_cost
                    )
                    / total_revenue * 100, 2
                ),
                "formula": (
                    "(Revenue - Cost) / Revenue * 100"
                ),
                "assumption": (
                    "Cost represents directly attributable "
                    "cost."
                )
            })

    if profit and revenue:
        total_revenue = safe_float(
            df[revenue].sum()
        )

        total_profit = safe_float(
            df[profit].sum()
        )

        if total_revenue:
            kpis.append({
                "name": "Profit margin",
                "value": round(
                    total_profit
                    / total_revenue * 100, 2
                ),
                "formula": (
                    "Total profit / total revenue * 100"
                ),
                "assumption": (
                    "Profit and revenue use consistent "
                    "definitions."
                )
            })

    if customer and order:
        unique_customers = df[customer].nunique()
        unique_orders = df[order].nunique()

        if unique_orders and unique_customers:
            kpis.append({
                "name": "Orders per customer",
                "value": round(
                    unique_orders
                    / unique_customers, 2
                ),
                "formula": (
                    "Unique orders / unique customers"
                ),
                "assumption": (
                    "Customer and order identifiers are "
                    "reliable."
                )
            })

    if date and revenue:
        parsed = pd.to_datetime(
            df[date], errors="coerce"
        ).dropna()

        if len(parsed) >= 2:
            kpis.append({
                "name": "Date coverage",
                "value": (
                    f"{parsed.min().date()} to "
                    f"{parsed.max().date()}"
                ),
                "formula": (
                    "Earliest to latest date"
                ),
                "assumption": (
                    "Date parsing succeeded for at least "
                    "two records."
                )
            })

    return kpis


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def generate_html_report(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    quality_score: float,
    quality_reasons: List[str]
) -> str:
    cleaning_rows = "".join(
        f"<tr><td>{item['time']}</td>"
        f"<td>{item['action']}</td>"
        f"<td>{item['details']}</td></tr>"
        for item in st.session_state.cleaning_log
    ) or (
        "<tr><td colspan='3'>No cleaning actions "
        "performed.</td></tr>"
    )

    column_rows = "".join(
        f"<tr><td>{p['column']}</td>"
        f"<td>{p['type']}</td>"
        f"<td>{p['missing']}</td>"
        f"<td>{p['unique']}</td></tr>"
        for p in profile["column_profiles"]
    )

    kpis = discover_kpis(
        df, profile["types"]
    )

    kpi_rows = "".join(
        f"<tr><td>{k['name']}</td>"
        f"<td>{k['value']}</td>"
        f"<td>{k['formula']}</td>"
        f"<td>{k['assumption']}</td></tr>"
        for k in kpis
    ) or (
        "<tr><td colspan='4'>No reliable KPIs "
        "could be calculated.</td></tr>"
    )

    return f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>AI Data Analysis Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; color:#0f172a; margin:40px; }}
            h1 {{ color:#0f172a; }}
            h2 {{ color:#155e75; border-bottom:2px solid #e2e8f0; padding-bottom:6px; }}
            table {{ width:100%; border-collapse:collapse; margin:12px 0; }}
            th, td {{ border:1px solid #cbd5e5; padding:8px; text-align:left; font-size:13px; }}
            th {{ background:#f1f5f9; }}
            .muted {{ color:#64748b; }}
        </style>
    </head>
    <body>
        <h1>AI Data Analysis Report</h1>
        <p class="muted">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        </p>

        <h2>Overview</h2>
        <p>
            <b>Rows:</b> {profile['rows']} |
            <b>Columns:</b> {profile['columns']} |
            <b>Duplicate rows:</b> {profile['duplicate_rows']}
        </p>
        <p><b>Data-quality score:</b> {quality_score}/100</p>
        <ul>
            {''.join(f'<li>{r}</li>' for r in quality_reasons)}
        </ul>

        <h2>Column profile</h2>
        <table>
            <tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr>
            {column_rows}
        </table>

        <h2>Cleaning log</h2>
        <table>
            <tr><th>Time</th><th>Action</th><th>Details</th></tr>
            {cleaning_rows}
        </table>

        <h2>Discovered KPIs</h2>
        <table>
            <tr><th>KPI</th><th>Value</th><th>Formula</th><th>Assumption</th></tr>
            {kpi_rows}
        </table>

        <h2>Limitations</h2>
        <ul>
            <li>Correlation does not establish causation.</li>
            <li>KPI calculations depend on the meaning and quality of source columns.</li>
            <li>Anomalies are statistical deviations, not necessarily errors.</li>
            <li>AI-generated explanations should be reviewed before business decisions.</li>
        </ul>
    </body>
    </html>
    """


def generate_pdf_report(
    df: pd.DataFrame,
    profile: Dict[str, Any],
    quality_score: float,
    quality_reasons: List[str]
) -> Optional[bytes]:
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    pdf = canvas.Canvas(
        buffer, pagesize=A4
    )

    width, height = A4
    y = height - 55

    def write(
        text: str,
        size: int = 11,
        heading: bool = False
    ):
        nonlocal y

        if y < 70:
            pdf.showPage()
            y = height - 55

        pdf.setFont(
            "Helvetica-Bold"
            if heading else "Helvetica",
            size
        )

        for line in simpleSplit(
            text,
            "Helvetica-Bold"
            if heading else "Helvetica",
            size,
            width - 80
        ):
            pdf.drawString(45, y, line)
            y -= size + 5

    write(
        "AI Data Analysis Report", 20, True
    )

    write(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        10
    )

    y -= 12

    write("Overview", 15, True)

    write(
        f"Rows: {profile['rows']} | "
        f"Columns: {profile['columns']} | "
        f"Duplicate rows: "
        f"{profile['duplicate_rows']}"
    )

    write(
        f"Data-quality score: "
        f"{quality_score}/100"
    )

    for reason in quality_reasons:
        write(f"- {reason}", 10)

    y -= 12
    write("Column profile", 15, True)

    for p in profile["column_profiles"][:30]:
        write(
            f"{p['column']} | {p['type']} | "
            f"missing: {p['missing']} | "
            f"unique: {p['unique']}",
            9
        )

    y -= 12
    write("Cleaning log", 15, True)

    if st.session_state.cleaning_log:
        for item in st.session_state.cleaning_log[-20:]:
            write(
                f"{item['time']} | "
                f"{item['action']} | "
                f"{item['details']}",
                9
            )

    else:
        write(
            "No cleaning actions performed.", 9
        )

    pdf.save()
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def render_hero():
    st.markdown(
        """
        <div class="app-hero">
            <h1>📊 AI Data Analysis</h1>
            <p>
                Upload data, understand its quality, explore visualizations,
                ask questions, detect anomalies, and generate evidence-based reports.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def metric_card(label: str, value: Any):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_dataset_status() -> bool:
    if st.session_state.df is None:
        st.info(
            "Upload a CSV or Excel file in the "
            "**Dashboard / Overview** workspace to begin."
        )

        return False

    return True


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

def main():
    st.title("AI Data Analysis")
    render_hero()

    groq_client = get_groq_client()

    with st.sidebar:
        st.header("Dataset")

        uploaded_file = st.file_uploader(
            "Upload CSV or Excel",
            type=["csv", "xlsx", "xls"],
            help=(
                "The original file is preserved; all "
                "transformations are applied to a working copy."
            )
        )

        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.getvalue()

                if (
                        st.session_state.file_name
                        != uploaded_file.name
                ):
                    with st.spinner(
                            "Loading and profiling dataset..."
                    ):
                        raw_df = load_dataframe(
                            file_bytes,
                            uploaded_file.name
                        )

                        raw_df, rename_notes = (
                            normalize_columns(raw_df)
                        )

                        issues = validate_dataframe(
                            raw_df
                        )

                        if issues and len(raw_df) == 0:
                            st.error(
                                "\n".join(issues)
                            )

                        else:
                            st.session_state.file_name = (
                                uploaded_file.name
                            )

                            st.session_state.df_original = (
                                raw_df.copy()
                            )

                            st.session_state.df = raw_df.copy()
                            st.session_state.cleaning_log = []
                            st.session_state.chat_history = []
                            st.session_state.analysis_context = {}
                            st.session_state.last_result = None
                            st.session_state.last_chart = None

                            for note in rename_notes:
                                add_cleaning_log(
                                    "Column normalization",
                                    note
                                )

                            st.success(
                                f"Loaded {uploaded_file.name}: "
                                f"{len(raw_df):,} rows, "
                                f"{len(raw_df.columns)} columns."
                            )

            except Exception as exc:
                st.error(
                    f"Could not load file: {exc}"
                )

        if st.session_state.df is not None:
            st.success(
                f"**Active dataset:** "
                f"{st.session_state.file_name}"
            )

            st.caption(
                f"{len(st.session_state.df):,} rows × "
                f"{len(st.session_state.df.columns)} columns"
            )

        st.divider()
        st.header("Groq AI")

        if groq_client:
            st.success("Groq API connected.")
        else:
            st.warning(
                "Groq API key is missing. Add it to your local "
                "`.env` file or Streamlit Secrets as "
                "`GROQ_API_KEY`."
            )

        available_models = get_available_groq_models(
            groq_client
        )

        groq_model = st.selectbox(
            "Groq model",
            available_models,
            index=0,
            help=(
                "Only models available to your Groq API key are "
                "shown. Choose a non-Enterprise model if you are "
                "using a free or developer Groq account."
            )
        )

        if st.button("Test Groq connection"):
            if groq_client:
                try:
                    response = groq_chat(
                        client=groq_client,
                        model=groq_model,
                        system_prompt=(
                            "Reply with OK."
                        ),
                        user_prompt="Test connection.",
                        temperature=0,
                        max_tokens=10
                    )

                    st.session_state.groq_status = "connected"
                    st.success(
                        f"Groq connected. Model response: "
                        f"{response}"
                    )

                except Exception as exc:
                    st.session_state.groq_status = "unavailable"
                    st.error(
                        f"Groq connection failed: {exc}"
                    )

            else:
                st.session_state.groq_status = "unavailable"
                st.warning(
                    "Add `GROQ_API_KEY` to your local `.env` file "
                    "or Streamlit Secrets first."
                )

        st.divider()

        if st.button(
                "Clear conversation context",
                use_container_width=True
        ):
            st.session_state.chat_history = []
            st.session_state.analysis_context = {}
            st.success(
                "Conversation context cleared."
            )

        st.divider()
        st.caption(
            "Built with Python, Streamlit, Pandas, NumPy, SciPy, "
            "Plotly, and Groq AI."
        )

    tabs = st.tabs([
        "📊 Dashboard", "🔍 Data Profiling",
        "🧹 Data Cleaning", "📈 EDA",
        "💡 Visualization Recommendations",
        "🎨 Visualization Explorer",
        "💬 Ask Your Data",
        "🚨 Anomaly Detection",
        "🎯 Business KPIs", "📄 Reports"
    ])

    df = st.session_state.df

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------
    with tabs[0]:
        if not render_dataset_status():
            st.markdown("### Supported uploads")
            st.markdown(
                "- CSV files with common encodings\n"
                "- Excel `.xlsx` and `.xls` files\n"
                "- Automatic duplicate-column handling and "
                "type inference"
            )

            st.stop()

        types = infer_column_types(df)
        profile = profile_dataframe(df)

        quality_score, quality_reasons = (
            calculate_quality_score(df, profile)
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            metric_card(
                "Rows", f"{profile['rows']:,}"
            )

        with col2:
            metric_card(
                "Columns", profile["columns"]
            )

        with col3:
            metric_card(
                "Missing values",
                f"{profile['total_missing']:,}"
            )

        with col4:
            metric_card(
                "Duplicate rows",
                f"{profile['duplicate_rows']:,}"
            )

        st.markdown("### Dataset quality")

        st.markdown(
            f"**Score:** {quality_score}/100 — "
            f"{quality_label(quality_score)}",
            unsafe_allow_html=True
        )

        with st.expander(
            "How this score is calculated"
        ):
            st.markdown(
                "This is an app-specific transparency score, "
                "not an industry standard. It starts at 100 and "
                "subtracts points for missing values, duplicate "
                "rows, constant columns, high-cardinality "
                "categorical fields, and numeric values stored "
                "as text."
            )

            for reason in quality_reasons:
                st.markdown(
                    f"- {reason}"
                )

        st.markdown("### Sample records")
        st.dataframe(
            df.head(10), use_container_width=True
        )

        st.markdown("### Quick column overview")

        overview = pd.DataFrame([
            {
                "Column": p["column"],
                "Type": p["type"],
                "Missing": p["missing"],
                "Missing %": p["missing_pct"],
                "Unique": p["unique"]
            }
            for p in profile["column_profiles"]
        ])

        st.dataframe(
            overview,
            use_container_width=True,
            hide_index=True
        )

    # ------------------------------------------------------------------
    # Data profiling
    # ------------------------------------------------------------------
    with tabs[1]:
        if not render_dataset_status():
            st.stop()

        profile = profile_dataframe(df)
        types = profile["types"]

        st.subheader("Automated data profile")
        st.caption(
            "Types are inferred from Pandas dtypes and value "
            "patterns. Review inferred types before relying on "
            "them for critical analysis."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            metric_card(
                "Numeric columns",
                sum(
                    1 for t in types.values()
                    if t == "numeric"
                )
            )

        with col2:
            metric_card(
                "Categorical columns",
                sum(
                    1 for t in types.values()
                    if t == "categorical"
                )
            )

        with col3:
            metric_card(
                "Datetime columns",
                sum(
                    1 for t in types.values()
                    if t == "datetime"
                )
            )

        for p in profile["column_profiles"]:
            with st.expander(
                f"**{p['column']}** — {p['type']}"
            ):
                col_a, col_b, col_c = st.columns(3)

                with col_a:
                    metric_card(
                        "Missing",
                        f"{p['missing']:,} "
                        f"({p['missing_pct']}%)"
                    )

                with col_b:
                    metric_card(
                        "Unique",
                        f"{p['unique']:,}"
                    )

                with col_c:
                    metric_card(
                        "Type",
                        p["type"].title()
                    )

                if p["type"] == "numeric":
                    st.markdown(
                        f"**Mean:** "
                        f"{format_number(p.get('mean'))} &nbsp; "
                        f"**Median:** "
                        f"{format_number(p.get('median'))} &nbsp; "
                        f"**Std:** "
                        f"{format_number(p.get('std'))} &nbsp; "
                        f"**Min:** "
                        f"{format_number(p.get('min'))} &nbsp; "
                        f"**Max:** "
                        f"{format_number(p.get('max'))} &nbsp; "
                        f"**Skewness:** "
                        f"{format_number(p.get('skewness'))}"
                    )

                elif p["type"] == "datetime":
                    st.markdown(
                        f"**Range:** "
                        f"{p.get('min_date')} to "
                        f"{p.get('max_date')} &nbsp; "
                        f"**Invalid dates:** "
                        f"{p.get('invalid_dates', 0)}"
                    )

                else:
                    st.markdown(
                        "**Most frequent values:**"
                    )

                    st.json(
                        p.get("top_values", {})
                    )

        if profile["constant_columns"]:
            st.warning(
                f"Constant or near-constant columns: "
                f"{', '.join(profile['constant_columns'])}"
            )

        if profile["high_cardinality_columns"]:
            st.warning(
                f"High-cardinality categorical columns: "
                f"{', '.join(profile['high_cardinality_columns'][:10])}"
            )

    # ------------------------------------------------------------------
    # Data cleaning
    # ------------------------------------------------------------------
    with tabs[2]:
        if not render_dataset_status():
            st.stop()

        st.subheader("Data quality and cleaning")
        st.caption(
            "The original uploaded dataset remains unchanged. "
            "Cleaning is applied to a separate working copy and "
            "every action is logged."
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if st.button(
                "Remove duplicate rows",
                use_container_width=True
            ):
                st.session_state.df = (
                    remove_duplicate_rows(
                        st.session_state.df
                    )
                )

                st.success(
                    "Duplicate rows removed."
                )

        with col2:
            clean_column = st.selectbox(
                "Column",
                st.session_state.df.columns,
                key="clean_col"
            )

        with col3:
            clean_action = st.selectbox(
                "Action",
                [
                    "Impute missing",
                    "Drop missing rows",
                    "Convert type",
                    "Standardize text"
                ],
                key="clean_action"
            )

        with col4:
            if clean_action == "Impute missing":
                method = st.selectbox(
                    "Method",
                    [
                        "Mean", "Median", "Mode",
                        "Custom value"
                    ],
                    key="impute_method"
                )

                custom_value = (
                    st.text_input(
                        "Custom value",
                        key="custom_value"
                    )
                    if method == "Custom value"
                    else None
                )

            else:
                method, custom_value = None, None

        if st.button(
            "Apply cleaning action",
            type="primary",
            use_container_width=True
        ):
            try:
                if clean_action == "Impute missing":
                    st.session_state.df = (
                        impute_missing_values(
                            st.session_state.df,
                            clean_column,
                            method,
                            custom_value
                        )
                    )

                elif clean_action == "Drop missing rows":
                    st.session_state.df = drop_missing_rows(
                        st.session_state.df,
                        clean_column
                    )

                elif clean_action == "Convert type":
                    target = st.selectbox(
                        "Target type",
                        [
                            "Numeric", "Datetime",
                            "Categorical", "Boolean"
                        ],
                        key="target_type"
                    )

                    st.session_state.df = (
                        convert_column_type(
                            st.session_state.df,
                            clean_column,
                            target
                        )
                    )

                elif clean_action == "Standardize text":
                    st.session_state.df = standardize_text(
                        st.session_state.df,
                        clean_column
                    )

                st.success(
                    "Cleaning action applied."
                )

            except Exception as exc:
                st.error(
                    f"Cleaning failed: {exc}"
                )

        st.markdown("### Outlier review")

        numeric_cols = [
            c for c, t in infer_column_types(
                st.session_state.df
            ).items()
            if t == "numeric"
        ]

        if numeric_cols:
            outlier_column = st.selectbox(
                "Numeric column",
                numeric_cols,
                key="outlier_column"
            )

            outlier_method = st.selectbox(
                "Method",
                ["IQR", "Z-score"],
                key="outlier_method"
            )

            if st.button(
                "Detect outliers",
                use_container_width=True
            ):
                series = st.session_state.df[
                    outlier_column
                ]

                mask = (
                    detect_outliers_iqr(series)
                    if outlier_method == "IQR"
                    else detect_outliers_zscore(
                        series
                    )
                )

                count = int(mask.sum())

                st.info(
                    f"{count} potential outliers detected using "
                    f"{outlier_method}. Outliers are statistical "
                    "deviations, not necessarily errors."
                )

                if count:
                    st.dataframe(
                        st.session_state.df.loc[mask].head(
                            50
                        ),
                        use_container_width=True
                    )

        st.markdown("### Cleaning log")

        if st.session_state.cleaning_log:
            st.dataframe(
                pd.DataFrame(
                    st.session_state.cleaning_log
                ),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "No cleaning actions have been applied yet."
            )

    # ------------------------------------------------------------------
    # EDA
    # ------------------------------------------------------------------
    with tabs[3]:
        if not render_dataset_status():
            st.stop()

        st.subheader(
            "Exploratory data analysis"
        )

        types = infer_column_types(df)

        numeric = [
            c for c, t in types.items()
            if t == "numeric"
        ]

        categorical = [
            c for c, t in types.items()
            if t == "categorical"
        ]

        st.markdown(
            "### Descriptive statistics"
        )

        if numeric:
            st.dataframe(
                df[numeric].describe().T,
                use_container_width=True
            )

        else:
            st.info(
                "No numeric columns are available for "
                "descriptive statistics."
            )

        st.markdown(
            "### Categorical frequencies"
        )

        if categorical:
            selected = st.selectbox(
                "Categorical column",
                categorical,
                key="eda_categorical"
            )

            frequencies = df[selected].value_counts(
                dropna=False
            ).reset_index()

            frequencies.columns = [
                selected, "Count"
            ]

            st.dataframe(
                frequencies.head(25),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "No categorical columns detected."
            )

        st.markdown("### Correlations")

        if len(numeric) >= 2:
            corr = df[numeric].corr(
                numeric_only=True
            )

            fig = px.imshow(
                corr, text_auto=".2f",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1,
                title="Correlation heatmap"
            )

            st.plotly_chart(
                fig, use_container_width=True
            )

            st.caption(
                "Correlation describes association, not "
                "causation."
            )

        else:
            st.info(
                "At least two numeric columns are needed for "
                "correlation analysis."
            )

        st.markdown(
            "### Automated observations"
        )

        observations = []

        for column in numeric[:8]:
            series = df[column].dropna()

            if len(series):
                skew = series.skew()

                if abs(skew) > 1:
                    direction = (
                        "right-skewed"
                        if skew > 0
                        else "left-skewed"
                    )

                    observations.append(
                        f"**{column}** is {direction} "
                        f"(skewness {skew:.2f})."
                    )

                outliers = int(
                    detect_outliers_iqr(
                        series
                    ).sum()
                )

                if outliers:
                    observations.append(
                        f"**{column}** contains "
                        f"{outliers} potential IQR "
                        "outliers."
                    )

        for column in categorical[:5]:
            top_value = (
                df[column].value_counts().idxmax()
                if df[column].notna().any()
                else None
            )

            if top_value is not None:
                share = (
                    df[column].value_counts(
                        normalize=True
                    ).max() * 100
                )

                if share > 80:
                    observations.append(
                        f"**{column}** is dominated by "
                        f"'{top_value}' ({share:.1f}% of "
                        "records)."
                    )

        if observations:
            for observation in observations:
                st.markdown(
                    f"<div class='insight-box'>"
                    f"{observation}</div>",
                    unsafe_allow_html=True
                )

        else:
            st.info(
                "No strong automated observations detected "
                "in the current dataset."
            )

    # ------------------------------------------------------------------
    # Visualization recommendations
    # ------------------------------------------------------------------
    with tabs[4]:
        if not render_dataset_status():
            st.stop()

        st.subheader(
            "Intelligent visualization recommendations"
        )

        types = infer_column_types(df)

        recommendations = recommend_visualizations(
            df, types
        )

        st.markdown("### Best visualizations")

        for recommendation in recommendations[:6]:
            with st.container():
                st.markdown(
                    f"""
                    <div class='insight-box'>
                    <b>{recommendation['chart']}</b> — suitability
                    {recommendation['score']}/100<br>
                    <span class='small-muted'>
                    {recommendation['reason']}
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if (
                    recommendation["x"]
                    or recommendation["y"]
                ):
                    try:
                        fig = build_chart(
                            df,
                            recommendation["chart"],
                            recommendation["x"],
                            recommendation["y"],
                            title=(
                                f"{recommendation['chart']}: "
                                f"{recommendation['y'] or ''} "
                                f"{recommendation['x'] or ''}"
                            ).strip()
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True
                        )

                    except Exception as exc:
                        st.warning(
                            f"Could not render "
                            f"{recommendation['chart']}: "
                            f"{exc}"
                        )

        st.markdown(
            "### Other available charts"
        )

        with st.expander(
            "Show all chart options and suitability"
        ):
            for recommendation in recommendations[6:]:
                suitable, reason = chart_is_suitable(
                    recommendation["chart"],
                    df,
                    types,
                    recommendation["x"],
                    recommendation["y"]
                )

                status = (
                    "✅" if suitable else "⚠️"
                )

                st.markdown(
                    f"- {status} "
                    f"**{recommendation['chart']}** — "
                    f"{recommendation['reason']}"
                )

    # ------------------------------------------------------------------
    # Interactive visualization explorer
    # ------------------------------------------------------------------
    with tabs[5]:
        if not render_dataset_status():
            st.stop()

        st.subheader(
            "Interactive visualization explorer"
        )

        st.caption(
            "Select any columns. The app automatically handles "
            "dates, text, numeric text, missing values, negative "
            "bubble sizes, and aggregation."
        )

        types = infer_column_types(df)

        numeric = [
            c for c, t in types.items()
            if t == "numeric"
        ]

        categorical = [
            c for c, t in types.items()
            if t == "categorical"
        ]

        dates = [
            c for c, t in types.items()
            if t == "datetime"
        ]

        col1, col2 = st.columns([1, 2])

        with col1:
            chart = st.selectbox(
                "Chart type",
                CHART_CATALOG,
                key="explorer_chart"
            )

            suitable, reason = chart_is_suitable(
                chart, df, types, None, None
            )

            if not suitable:
                st.info(reason)

            x_options = [
                "None"
            ] + list(df.columns)

            y_options = [
                "None"
            ] + list(df.columns)

            color_options = [
                "None"
            ] + list(df.columns)

            size_options = [
                "None"
            ] + list(df.columns)

            x = st.selectbox(
                "X / category",
                x_options,
                key="explorer_x"
            )

            y = st.selectbox(
                "Y / metric",
                y_options,
                key="explorer_y"
            )

            color = st.selectbox(
                "Color / group",
                color_options,
                key="explorer_color"
            )

            size = st.selectbox(
                "Bubble size",
                size_options,
                key="explorer_size"
            )

            aggregation = st.selectbox(
                "Aggregation",
                [
                    "Sum", "Mean", "Median",
                    "Count", "Minimum", "Maximum",
                    "None"
                ],
                key="explorer_agg"
            )

            title = st.text_input(
                "Custom title",
                key="explorer_title"
            )

            if chart in [
                "Line", "Area", "Moving average",
                "Multi-line"
            ] and dates:
                st.session_state.analysis_context[
                    "moving_average_window"
                ] = st.slider(
                    "Moving-average window",
                    2, 60, 7,
                    key="moving_average_window"
                )

        with col2:
            if st.button(
                "Generate visualization",
                type="primary",
                use_container_width=True
            ):
                try:
                    x_column = (
                        None if x == "None" else x
                    )

                    y_column = (
                        None if y == "None" else y
                    )

                    color_column = (
                        None
                        if color == "None"
                        else color
                    )

                    size_column = (
                        None
                        if size == "None"
                        else size
                    )

                    fig = build_chart(
                        df=df,
                        chart=chart,
                        x=x_column,
                        y=y_column,
                        color=color_column,
                        size=size_column,
                        aggregation=aggregation,
                        title=title or None
                    )

                    st.session_state.last_chart = fig

                    st.session_state.analysis_context[
                        "last_chart"
                    ] = {
                        "chart": chart,
                        "x": x_column,
                        "y": y_column,
                        "color": color_column,
                        "size": size_column,
                        "aggregation": aggregation
                    }

                except Exception as exc:
                    st.error(
                        f"Could not create chart: {exc}"
                    )

            if (
                st.session_state.last_chart
                is not None
            ):
                st.plotly_chart(
                    st.session_state.last_chart,
                    use_container_width=True
                )

                fig_buffer = io.BytesIO()

                try:
                    st.session_state.last_chart.write_image(
                        fig_buffer,
                        format="png",
                        scale=2
                    )

                    st.download_button(
                        "Download chart as PNG",
                        data=fig_buffer.getvalue(),
                        file_name="chart.png",
                        mime="image/png",
                        use_container_width=True
                    )

                except Exception:
                    st.caption(
                        "PNG export requires the optional "
                        "`kaleido` package. HTML export is "
                        "available below."
                    )

                html_buffer = io.StringIO()

                st.session_state.last_chart.write_html(
                    html_buffer,
                    include_plotlyjs="cdn"
                )

                st.download_button(
                    "Download chart as HTML",
                    data=html_buffer.getvalue(),
                    file_name="chart.html",
                    mime="text/html",
                    use_container_width=True
                )

    # ------------------------------------------------------------------
    # Ask Your Data / General AI Assistant
    # ------------------------------------------------------------------
    with tabs[6]:
        st.subheader(
            "Ask Your Data / General AI Assistant"
        )

        st.caption(
            "Ask anything. Dataset questions use verified "
            "calculations; general questions use Groq AI."
        )

        if not render_dataset_status():
            st.info(
                "You can ask general analytics, business, "
                "statistics, Python, or AI questions. Upload a "
                "dataset to enable dataset-specific analysis."
            )

            has_data = False

        else:
            has_data = True

        for message in st.session_state.chat_history:
            role = message["role"]
            content = message["content"]

            css_class = (
                "chat-user"
                if role == "user"
                else "chat-ai"
            )

            st.markdown(
                f"<div class='{css_class}'><b>"
                f"{'You' if role == 'user' else 'Assistant'}"
                f"</b><br>{content}</div>",
                unsafe_allow_html=True
            )

            if message.get("chart") is not None:
                st.plotly_chart(
                    message["chart"],
                    use_container_width=True
                )

        question = st.chat_input(
            "Ask about your data, request a chart, or ask any "
            "general question"
        )

        if question:
            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            response = ""
            chart_figure = None

            try:
                types = (
                    infer_column_types(df)
                    if has_data else {}
                )

                intent = classify_intent(
                    question, has_data
                )

                dataset_answered = False

                if has_data and intent in [
                    "dataset_analytical",
                    "dataset_visualization",
                    "follow_up"
                ]:
                    plan = None

                    if groq_client:
                        plan = extract_plan_with_groq(
                            question=question,
                            df=df,
                            types=types,
                            client=groq_client,
                            model=groq_model
                        )

                    if plan is None:
                        plan = rule_based_plan(
                            question,
                            df,
                            types,
                            st.session_state.analysis_context
                        )

                    result = execute_analytical_plan(
                        df, plan
                    )

                    if "error" not in result:
                        if result.get("column"):
                            st.session_state.analysis_context[
                                "last_metric"
                            ] = result["column"]

                        if result.get("group_by"):
                            st.session_state.analysis_context[
                                "last_category"
                            ] = result["group_by"]

                        if result.get("top_item"):
                            item = result["top_item"]
                            key = list(item.keys())[0]

                            st.session_state.analysis_context[
                                "last_top_value"
                            ] = item.get(key)

                        finding = ""

                        if result.get("value") is not None:
                            finding = (
                                f"The calculated "
                                f"{result.get('column', 'value')} "
                                f"is **{format_number(result['value'])}**."
                            )

                        elif result.get("top_item"):
                            item = result["top_item"]
                            key = list(item.keys())[0]

                            finding = (
                                f"**{item[key]}** has the highest "
                                f"{result.get('column', 'metric')}: "
                                f"**{format_number(item.get('value'))}**."
                            )

                        elif result.get("table") is not None:
                            finding = (
                                f"The analysis covers "
                                f"**{result.get('rows_analyzed', 0):,} "
                                "rows**."
                            )

                        response = f"""**Finding:** {finding}

**Evidence:** The calculation used {result.get('rows_analyzed', 0):,} rows after applying validated filters.

**Interpretation:** This is an observed result from the current dataset. It does not by itself establish causation.

**Recommendation:** Review the supporting table and chart, then validate the result against business context before acting.

**Method:** Python/Pandas performed the calculation. Groq AI proposed a constrained analysis plan."""

                        if result.get("table") is not None:
                            with st.expander(
                                "Evidence: result table"
                            ):
                                st.dataframe(
                                    result["table"],
                                    use_container_width=True,
                                    hide_index=True
                                )

                        chart_type = result.get("chart")

                        if chart_type:
                            try:
                                chart_figure = build_chart(
                                    df=df,
                                    chart=chart_type,
                                    x=result.get("chart_x"),
                                    y=(
                                        result.get("chart_y")
                                        or result.get("column")
                                    ),
                                    title=question[:70]
                                )

                            except Exception as exc:
                                st.caption(
                                    f"Chart could not be generated: "
                                    f"{exc}"
                                )

                        insight = generate_ai_insight(
                            question=question,
                            result=result,
                            client=groq_client,
                            model=groq_model
                        )

                        response += (
                            f"\n\n**AI interpretation:**\n\n"
                            f"{insight}"
                        )

                        dataset_answered = True

                if not dataset_answered:
                    response = answer_general_question(
                        question=question,
                        client=groq_client,
                        model=groq_model,
                        df=df if has_data else None,
                        has_data=has_data
                    )

            except Exception as exc:
                response = (
                    f"**I encountered an issue while processing "
                    f"that request:** {exc}. Please try rephrasing "
                    "the question or selecting different columns."
                )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response,
                "chart": chart_figure
            })

            st.rerun()

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------
    with tabs[7]:
        if not render_dataset_status():
            st.stop()

        st.subheader("Anomaly detection")
        st.caption(
            "Anomalies are statistical deviations. They may indicate "
            "errors, rare events, fraud, or genuinely important "
            "business cases."
        )

        types = infer_column_types(df)

        numeric = [
            c for c, t in types.items()
            if t == "numeric"
        ]

        if not numeric:
            st.info(
                "Anomaly detection requires at least one numeric "
                "column."
            )

            st.stop()

        col1, col2, col3 = st.columns(3)

        with col1:
            anomaly_column = st.selectbox(
                "Numeric column",
                numeric,
                key="anomaly_column"
            )

        with col2:
            method = st.selectbox(
                "Method",
                [
                    "IQR", "Z-score",
                    "Isolation Forest"
                ],
                key="anomaly_method"
            )

        with col3:
            threshold = st.slider(
                "Threshold", 1.0, 5.0, 3.0, 0.1,
                key="anomaly_threshold"
            )

        if st.button(
            "Detect anomalies",
            type="primary",
            use_container_width=True
        ):
            with st.spinner(
                "Detecting anomalies..."
            ):
                series = df[anomaly_column]

                if method == "IQR":
                    mask = detect_outliers_iqr(
                        series
                    )

                    parameters = (
                        "IQR multiplier: 1.5"
                    )

                elif method == "Z-score":
                    mask = detect_outliers_zscore(
                        series, threshold
                    )

                    parameters = (
                        f"Absolute Z-score threshold: "
                        f"{threshold}"
                    )

                else:
                    if not SKLEARN_AVAILABLE:
                        st.error(
                            "Isolation Forest requires "
                            "scikit-learn. Install it with "
                            "`pip install scikit-learn`."
                        )

                        mask = pd.Series(
                            False, index=df.index
                        )

                        parameters = "Unavailable"

                    else:
                        values = (
                            pd.to_numeric(
                                series,
                                errors="coerce"
                            )
                            .fillna(series.median())
                            .values.reshape(-1, 1)
                        )

                        model = IsolationForest(
                            contamination=0.05,
                            random_state=42
                        )

                        mask = pd.Series(
                            model.fit_predict(values) == -1,
                            index=df.index
                        )

                        parameters = (
                            "Isolation Forest contamination: "
                            "5%"
                        )

                count = int(mask.sum())

                st.info(
                    f"**{count} anomalies detected** using "
                    f"{method}. {parameters}"
                )

                if count:
                    anomaly_df = df.loc[mask].copy()

                    anomaly_df["anomaly_score"] = np.abs(
                        (
                            pd.to_numeric(
                                anomaly_df[
                                    anomaly_column
                                ],
                                errors="coerce"
                            )
                            - pd.to_numeric(
                                df[anomaly_column],
                                errors="coerce"
                            ).mean()
                        )
                        / (
                            pd.to_numeric(
                                df[anomaly_column],
                                errors="coerce"
                            ).std() or 1
                        )
                    )

                    st.dataframe(
                        anomaly_df.head(100),
                        use_container_width=True
                    )

                    fig = px.scatter(
                        df.reset_index(),
                        x=df.reset_index().index,
                        y=anomaly_column,
                        color=mask.reset_index(
                            drop=True
                        ),
                        color_discrete_map={
                            True: "red",
                            False: "#94a3b8"
                        },
                        title=(
                            f"Anomaly detection: "
                            f"{anomaly_column}"
                        )
                    )

                    st.plotly_chart(
                        fig, use_container_width=True
                    )

    # ------------------------------------------------------------------
    # Business KPIs
    # ------------------------------------------------------------------
    with tabs[8]:
        if not render_dataset_status():
            st.stop()

        st.subheader(
            "Automatic business KPI discovery"
        )

        st.caption(
            "KPIs are proposed only when the required columns and "
            "formulas can be reliably computed."
        )

        types = infer_column_types(df)

        kpis = discover_kpis(df, types)

        if not kpis:
            st.info(
                "No reliable KPIs could be inferred from the "
                "current column names and data types."
            )

        else:
            cols = st.columns(3)

            for index, kpi in enumerate(kpis):
                with cols[index % 3]:
                    value = kpi["value"]

                    display_value = (
                        format_number(value)
                        if isinstance(
                            value,
                            (
                                int, float,
                                np.floating,
                                np.integer
                            )
                        )
                        else value
                    )

                    metric_card(
                        kpi["name"], display_value
                    )

                    with st.expander(
                        "Formula and assumptions"
                    ):
                        st.markdown(
                            f"**Formula:** "
                            f"{kpi['formula']}\n\n"
                            f"**Assumption:** "
                            f"{kpi['assumption']}"
                        )

        st.markdown("### Important limitation")
        st.markdown(
            """
            <div style="
                background-color: #000000;
                color: #ffffff;
                padding: 1rem 1.25rem;
                border-radius: 10px;
                border: 1px solid #ffffff;
                font-size: 1rem;
                line-height: 1.55;
            ">
                <strong>Important limitation</strong><br>
                KPI inference is based on column names and data structure.
                Confirm that fields such as revenue, cost, customer, and
                order represent the intended business definitions before
                using results for decisions.
            </div>
            """,
            unsafe_allow_html=True
        )
    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------
    with tabs[9]:
        if not render_dataset_status():
            st.stop()

        st.subheader("Reports and downloads")

        profile = profile_dataframe(df)

        quality_score, quality_reasons = (
            calculate_quality_score(df, profile)
        )

        st.markdown("### Report preview")

        st.markdown(
            f"- **Dataset:** "
            f"{st.session_state.file_name}\n"
            f"- **Rows:** {profile['rows']:,}\n"
            f"- **Columns:** {profile['columns']}\n"
            f"- **Quality score:** "
            f"{quality_score}/100"
        )

        st.markdown("**Quality findings:**")

        for reason in quality_reasons:
            st.markdown(
                f"- {reason}"
            )

        st.markdown("### Downloads")
        col1, col2, col3 = st.columns(3)

        with col1:
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)

            st.download_button(
                "Download cleaned CSV",
                data=csv_buffer.getvalue(),
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True
            )

        with col2:
            html_report = generate_html_report(
                df,
                profile,
                quality_score,
                quality_reasons
            )

            st.download_button(
                "Download HTML report",
                data=html_report,
                file_name="ai_data_analysis_report.html",
                mime="text/html",
                use_container_width=True
            )

        with col3:
            pdf_bytes = generate_pdf_report(
                df,
                profile,
                quality_score,
                quality_reasons
            )

            if pdf_bytes:
                st.download_button(
                    "Download PDF report",
                    data=pdf_bytes,
                    file_name="ai_data_analysis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

            else:
                st.info(
                    "Install ReportLab to enable PDF export: "
                    "`pip install reportlab`."
                )

        st.markdown("### Full cleaning log")

        if st.session_state.cleaning_log:
            st.dataframe(
                pd.DataFrame(
                    st.session_state.cleaning_log
                ),
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info(
                "No cleaning actions have been recorded."
            )


if __name__ == "__main__":
    main()