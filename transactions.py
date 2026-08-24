"""Transactions — Robust Bank Transaction Categorization.

Canonical implementation for the controlled synthetic benchmark
``DoDataThings/us-bank-transaction-categories-v2``.

Central scientific question
---------------------------
Can transaction-categorization models learn generalizable transaction
semantics and banking-language patterns rather than simply memorizing
merchant identities and formatting artifacts?

The module provides:
  * ETL + schema validation for the DoDataThings v2 dataset
  * text-normalization experiment (raw / normalized / structure-aware)
  * word-, character- and hybrid TF-IDF representations (sparse)
  * structured feature engineering (direction + structural text stats)
  * logistic regression / linear SVM / XGBoost comparison
  * merchant-disjoint + transaction-format-disjoint generalization
  * merchant-memorization ablation
  * calibration + confidence-aware selective prediction
  * learning-curve (data efficiency)
  * error analysis
  * artifact / report generation (out/*.json, reports/*.md)

All models use RANDOM_STATE = 42 and are evaluated with the same splits.
This is a synthetic benchmark; results do not estimate bank production
performance.

Usage:
    python transactions.py --out ./out --reports ./reports
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("OMP_NUM_THREADS", "4")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("transactions")

RANDOM_STATE = 42
DATASET_ID = "DoDataThings/us-bank-transaction-categories-v2"

# Validated on load; approximate per the dataset card.
EXPECTED_CATEGORIES = 17

# Characters that are "structural" in banking descriptions.
_REF_ID_PATTERN = re.compile(r"(?i)\b(?:ppd|web)\s*id\s*[:=]\s*[\w.-]+")
_ID_PATTERN = re.compile(r"(?i)\bid\s*[:=]\s*[\w.-]+")
_ZIP_STATE_USA = re.compile(r"(?i)\b\d{5}(-\d{4})?\s*[a-z]{2}\s*usa\b")
_ORDER_ID = re.compile(r"\*\w+")
_STORE_NO = re.compile(r"#\d+")
_ADDRESS_TAIL = re.compile(r"(?i)\b\d{5}(-\d{4})?\s*[a-z]{2}\s*usa\b\s*$")


# ═══════════════════════════════════════════════════════════════════
#  Seed / logging helpers
# ═══════════════════════════════════════════════════════════════════

def seed_everything(seed: int = RANDOM_STATE) -> None:
    """Reproducibility — random, numpy, and hash seeds."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ═══════════════════════════════════════════════════════════════════
#  ETL — dataset loading + schema
# ═══════════════════════════════════════════════════════════════════

def load_raw_dataframe(
    dataset_id: str = DATASET_ID, max_rows: Optional[int] = None
) -> pd.DataFrame:
    """Load the DoDataThings v2 dataset and return a pandas DataFrame.

    ``max_rows`` is for fast smoke runs only; the canonical run uses None.
    """
    from datasets import load_dataset

    logger.info("Loading %s …", dataset_id)
    ds = load_dataset(dataset_id, split="train")
    if max_rows is not None:
        ds = ds.select(list(range(min(max_rows, len(ds)))))
    df = ds.to_pandas()  # type: ignore[assignment]
    logger.info("Loaded %d rows, columns=%s", len(df), list(df.columns))
    return df


def inspect_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """Schema table: column, dtype, missing %, unique values, role."""
    roles: Dict[str, str] = {
        "description": "transaction text",
        "category": "target category",
    }
    table = []
    for col in df.columns:
        n_missing = int(df[col].isna().sum())
        table.append(
            {
                "column": col,
                "dtype": str(df[col].dtype),
                "missing": n_missing,
                "missing_pct": round(100.0 * n_missing / len(df), 3),
                "unique": int(df[col].nunique()),
                "role": roles.get(col, "metadata"),
            }
        )
    return {"n_rows": int(len(df)), "columns": table}


# ═══════════════════════════════════════════════════════════════════
#  Field extraction — direction, merchant, format, city
# ═══════════════════════════════════════════════════════════════════

def extract_direction(description: str) -> str:
    """Return 'debit' / 'credit' from the leading [debit]/[credit] tag."""
    m = re.match(r"^\[\s*(debit|credit)\s*\]", str(description), flags=re.I)
    return m.group(1).lower() if m else "unknown"


def _strip_direction(description: str) -> str:
    return re.sub(r"^\[\s*(debit|credit)\s*\]\s*", "", str(description), flags=re.I)


def extract_merchant(description: str) -> str:
    """Heuristic merchant key.

    Removes direction tag, reference IDs, trailing US-address blocks, order
    IDs, and store numbers, then truncates at the first standalone digit
    token (street/amount-ish noise). Intentionally heuristic; documented in
    the leakage audit.
    """
    t = _strip_direction(description)
    t = _REF_ID_PATTERN.sub(" ", t)
    t = _ID_PATTERN.sub(" ", t)
    # trailing "94587 CA USA" / "27601-7390nc usa" blocks
    t = re.sub(r"(?i)\b\d{5}(-\d{4})?\s*[a-z]{2}\s*usa\b\s*$", " ", t)
    t = re.sub(r"(?i)\s*usa\b\s*$", " ", t)
    t = _ORDER_ID.sub(" ", t)
    t = _STORE_NO.sub(" ", t)
    # generic wrapper phrases
    for pat in [
        r"(?i)paypal inst xfer\s*",
        r"(?i)preapproved payment bill user payment:\s*",
        r"(?i)preapproved payment:\s*",
        r"(?i)express checkout payment:\s*",
        r"(?i)payment to:\s*",
        r"(?i)payment from\s+",
        r"(?i)withdrawal from\s+",
        r"(?i)preauthorized deposit from\s+",
        r"(?i)preauthorized withdrawal to\s+",
        r"(?i)rent pmt\s+",
        r"(?i)rent payment\s+",
        r"(?i)mtg pmt\s+",
        r"(?i)mortgage payment\s+",
        r"(?i)monthly installments\b.*$",
        r"(?i)orig co name:\s*",
        r"(?i)entry descr:\s*",
        r"(?i)sec:ppd\b",
    ]:
        t = re.sub(pat, " ", t)
    # POS wrapper prefixes
    t = re.sub(r"(?i)^(?:sq|tst|clv|pp|pypl)\s*\*\s*", " ", t)
    # truncate at first standalone digit token (street no., amount-ish) BEFORE
    # dropping digit tokens, so the merchant keeps only alpha head
    toks = t.split()
    cut = len(toks)
    for i, tok in enumerate(toks):
        if tok.isdigit() and len(tok) >= 3:
            cut = i
            break
    toks = toks[:cut]
    # drop punctuation-only tokens that remain
    toks = [tok for tok in toks if re.search(r"[a-zA-Z]", tok)]
    merchant = " ".join(toks).strip(" -:,;'\"")
    return merchant.lower() if merchant else "unknown_merchant"


def detect_format(description: str) -> str:
    """Classify the transaction-description format family.

    Order matters: specific wrapper families are checked before generic ones.
    """
    t = _strip_direction(description)
    lo = t.lower()
    # POS wrappers (Square/Toast/Clover) first — most specific prefix
    if re.match(r"^(sq|tst|clv)\s*\*", lo):
        return "pos"
    # PayPal wrapper family — includes PayPal ACH-style installments
    if re.match(r"^(pp|pypl|paypal)\s*\*", lo) or "paypal" in lo:
        return "paypal"
    if "preapproved" in lo or "express checkout" in lo or "paypal inst xfer" in lo:
        return "paypal"
    # Capital One style
    if re.search(r"withdrawal from|preauthorized deposit from", lo):
        return "capital_one"
    # ACH reference identifiers (PPD / WEB ID / ID:)
    if re.search(r"ppd\s*id|web\s*id|id\s*[:=]", lo):
        return "ach"
    # Address-style (Apple Card) descriptions
    if re.search(r"\b\d{5}(-\d{4})?\s*[a-z]{2}\s*usa\b", lo):
        return "address"
    return "simple"


def extract_city(description: str) -> Optional[str]:
    """Extract a US city from an address-style description, if present."""
    t = _strip_direction(description)
    # "CITY 94587 CA USA" with optional spaces, or mashed "CITY94587"
    m = re.search(r"([A-Za-z][A-Za-z .'-]{1,25}?)\s*(\d{5}(?:-\d{4})?)\s*[A-Z]{2}\s*USA\b", t, re.I)
    if m:
        city = m.group(1).strip()
        if " " in city:
            city = city.split()[-1]
        return city.lower()
    return None


def format_counts(df: pd.DataFrame) -> Dict[str, int]:
    """Count detected transaction formats."""
    return dict(Counter(df["format"].astype(str)))


# ═══════════════════════════════════════════════════════════════════
#  Text normalization experiment
# ═══════════════════════════════════════════════════════════════════

def normalize_raw(text: str) -> str:
    """Representation A — minimal: direction tag + whitespace collapse."""
    return re.sub(r"\s+", " ", _strip_direction(text)).strip().lower()


def normalize_basic(text: str) -> str:
    """Representation B — lowercase, collapse whitespace, keep structure."""
    return re.sub(r"\s+", " ", _strip_direction(text)).strip().lower()


def normalize_structured(text: str) -> str:
    """Representation C — mask variable identifiers.

    Numeric order/ref codes → <REF>, store numbers → <STORE>, card/account
    digit blocks → <NUM>.  Meaningful alphabetic tokens are preserved.
    """
    t = _strip_direction(text)
    t = _REF_ID_PATTERN.sub(" <REF> ", t)
    t = _ID_PATTERN.sub(" <REF> ", t)
    t = re.sub(r"\*\w+", " <REF>", t)
    t = re.sub(r"#\d+", " <STORE>", t)
    t = re.sub(r"\b\d{4,}\b", " <NUM> ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


# ═══════════════════════════════════════════════════════════════════
#  Structured features
# ═══════════════════════════════════════════════════════════════════

def structural_features(df: pd.DataFrame, text_col: str = "description") -> pd.DataFrame:
    """Lightweight text-structure features per row."""
    out = pd.DataFrame(index=df.index)
    texts = df[text_col].astype(str)
    out["char_count"] = texts.str.len()
    out["token_count"] = texts.str.split().str.len()
    out["digit_count"] = texts.str.count(r"\d")
    out["digit_ratio"] = np.where(
        out["char_count"] > 0, out["digit_count"] / out["char_count"], 0.0
    )
    out["punct_count"] = texts.str.count(r"[^a-zA-Z0-9\s]")
    out["alpha_token_count"] = texts.str.findall(r"\b[a-zA-Z]+\b").str.len()
    out["numeric_token_count"] = texts.str.findall(r"\b\d+\b").str.len()
    return out


def direction_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot [debit, credit] direction."""
    return pd.get_dummies(df["direction"], prefix="dir", dtype=float)


# ═══════════════════════════════════════════════════════════════════
#  TF-IDF representations (sparse)
# ═══════════════════════════════════════════════════════════════════

_WORD_PARAMS = dict(
    analyzer="word", ngram_range=(1, 2), max_features=15_000,
    sublinear_tf=True, min_df=2, max_df=0.95,
)
_CHAR_PARAMS = dict(
    analyzer="char", ngram_range=(3, 5), max_features=20_000,
    sublinear_tf=True, min_df=3, max_df=0.95,
)


def fit_word_tfidf(train_texts: Sequence[str], **kw: Any) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    params = dict(_WORD_PARAMS)
    params.update(kw)
    return TfidfVectorizer(**params).fit(train_texts)


def fit_char_tfidf(train_texts: Sequence[str], **kw: Any) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    params = dict(_CHAR_PARAMS)
    params.update(kw)
    return TfidfVectorizer(**params).fit(train_texts)


def vectorize_pair(vectorizer: Any, train_texts: Sequence[str], test_texts: Sequence[str]):
    Xtr = vectorizer.transform(train_texts)
    Xte = vectorizer.transform(test_texts)
    return Xtr, Xte


def build_feature_matrix(
    X_text: Any,
    df: pd.DataFrame,
    use_direction: bool = False,
    use_structural: bool = False,
) -> Any:
    """Stack sparse text features with optional direction / structural features."""
    from scipy.sparse import hstack

    mats = [X_text]
    if use_direction:
        mats.append(direction_onehot(df).astype(float).values)
    if use_structural:
        mats.append(structural_features(df, text_col="description").astype(float).values)
    if len(mats) == 1:
        return X_text
    return hstack(mats, format="csr")


# ═══════════════════════════════════════════════════════════════════
#  Models
# ═══════════════════════════════════════════════════════════════════

def train_logistic_regression(X: Any, y: np.ndarray, seed: int = RANDOM_STATE, **kw: Any) -> Any:
    from sklearn.linear_model import LogisticRegression

    params = dict(max_iter=8000, solver="lbfgs", random_state=seed)
    params.update(kw)
    return LogisticRegression(**params).fit(X, y)


def train_linear_svc(X: Any, y: np.ndarray, seed: int = RANDOM_STATE, **kw: Any) -> Any:
    from sklearn.svm import LinearSVC

    params = dict(max_iter=3000, random_state=seed)
    params.update(kw)
    return LinearSVC(**params).fit(X, y)


def train_xgboost(
    X: Any,
    y: np.ndarray,
    num_class: int,
    seed: int = RANDOM_STATE,
    n_estimators: int = 150,
    **kw: Any,
) -> Any:
    """XGBoost with label-encoded targets; returns a string-label wrapper."""
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    params = dict(
        objective="multi:softprob",
        num_class=num_class,
        n_estimators=n_estimators,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
    )
    params.update(kw)
    model = xgb.XGBClassifier(**params).fit(X, y_enc)
    return XGBStringWrapper(model, le)


class XGBStringWrapper:
    """XGBoost wrapper that maps integer predictions back to string labels."""

    def __init__(self, model: Any, le: Any) -> None:
        self._model = model
        self._le = le

    def predict(self, X: Any) -> np.ndarray:
        idx = self._model.predict(X)
        return self._le.inverse_transform(idx.astype(int))

    def predict_proba(self, X: Any) -> np.ndarray:
        return self._model.predict_proba(X)

    @property
    def classes_(self) -> np.ndarray:
        return self._le.classes_

    def get_params(self, deep: bool = True) -> dict:
        return self._model.get_params(deep)

    def __getstate__(self) -> dict:
        return {"_model": self._model, "_le": self._le}

    def __setstate__(self, state: dict) -> None:
        self._model = state["_model"]
        self._le = state["_le"]


# ═══════════════════════════════════════════════════════════════════
#  Evaluation
# ═══════════════════════════════════════════════════════════════════

def evaluate_model(
    model: Any,
    X: Any,
    y: np.ndarray,
    label_names: List[str],
    model_name: str,
) -> Dict[str, Any]:
    """Accuracy, macro-F1, weighted-F1, per-class F1, confusion matrix."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    y_pred = model.predict(X)
    acc = float(accuracy_score(y, y_pred))
    macro = float(f1_score(y, y_pred, average="macro", zero_division=0, labels=label_names))
    weighted = float(f1_score(y, y_pred, average="weighted", zero_division=0, labels=label_names))
    report = classification_report(
        y, y_pred, target_names=label_names, labels=label_names,
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y, y_pred, labels=label_names).tolist()
    logger.info("%s — Acc %.4f  Macro-F1 %.4f  W-F1 %.4f", model_name, acc, macro, weighted)
    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro, 4),
        "weighted_f1": round(weighted, 4),
        "classification_report": report,
        "confusion_matrix": cm,
    }


def make_splits(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["category"], random_state=seed
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def merchant_disjoint_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Grouped split: no merchant in test appears in train (zero overlap)."""
    from sklearn.model_selection import GroupShuffleSplit

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["merchant"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    overlap = len(set(train_df["merchant"]) & set(test_df["merchant"]))
    if overlap != 0:
        raise AssertionError(f"Merchant overlap should be 0, got {overlap}")
    return train_df, test_df, {"merchant_overlap": int(overlap)}


def merchant_overlap_fraction(train_df: pd.DataFrame, test_df: pd.DataFrame) -> float:
    train_merchants = set(train_df["merchant"])
    return float(test_df["merchant"].isin(train_merchants).mean())


# ═══════════════════════════════════════════════════════════════════
#  Classification pipeline helper
# ═══════════════════════════════════════════════════════════════════

def classify(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rep: str = "hybrid",
    model_name: str = "logreg",
    use_direction: bool = False,
    use_structural: bool = False,
    max_features_word: int = 15_000,
    max_features_char: int = 20_000,
    seed: int = RANDOM_STATE,
    n_estimators: int = 150,
) -> Tuple[Any, Dict[str, Any]]:
    """Vectorize (fit on train only) → train classifier → evaluate on test.

    ``rep`` ∈ {"word", "char", "hybrid"} selects the text representation.
    Returns (model, metrics).
    """
    label_names = sorted(train_df["category"].unique())
    y_train = train_df["category"].values
    y_test = test_df["category"].values

    wv = fv = None
    X_tr, X_te = None, None
    Xc_tr = Xc_te = None
    if rep in ("word", "hybrid"):
        wv = fit_word_tfidf(train_df["description"], max_features=max_features_word)
        X_tr, X_te = vectorize_pair(wv, train_df["description"], test_df["description"])
    if rep in ("char", "hybrid"):
        fv = fit_char_tfidf(train_df["description"], max_features=max_features_char)
        Xc_tr, Xc_te = vectorize_pair(fv, train_df["description"], test_df["description"])
        if rep == "char":
            X_tr, X_te = Xc_tr, Xc_te
    if rep == "hybrid":
        from scipy.sparse import hstack
        X_tr, X_te = hstack([X_tr, Xc_tr], format="csr"), hstack([X_te, Xc_te], format="csr")

    X_tr = build_feature_matrix(X_tr, train_df, use_direction, use_structural)
    X_te = build_feature_matrix(X_te, test_df, use_direction, use_structural)

    if model_name == "logreg":
        model = train_logistic_regression(X_tr, y_train, seed=seed)
    elif model_name == "linearsvc":
        model = train_linear_svc(X_tr, y_train, seed=seed)
    elif model_name == "xgboost":
        model = train_xgboost(X_tr, y_train, num_class=len(label_names), seed=seed,
                              n_estimators=n_estimators)
    else:
        raise ValueError(f"unknown model {model_name}")

    metrics = evaluate_model(model, X_te, y_test, label_names, f"{model_name} · {rep}")
    return model, metrics


# ═══════════════════════════════════════════════════════════════════
#  Feature ablation
# ═══════════════════════════════════════════════════════════════════

def feature_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Incremental feature-family ablation with LogisticRegression."""
    rows: Dict[str, Dict[str, float]] = {}
    for label, rep, ud, us in [
        ("word", "word", False, False),
        ("char", "char", False, False),
        ("word+char", "hybrid", False, False),
        ("word+char+dir", "hybrid", True, False),
        ("word+char+dir+struct", "hybrid", True, True),
    ]:
        _, m = classify(
            train_df, test_df, rep=rep, model_name="logreg",
            use_direction=ud, use_structural=us, seed=seed,
        )
        rows[label] = {
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "weighted_f1": m["weighted_f1"],
        }
    return rows


# ═══════════════════════════════════════════════════════════════════
#  Merchant memorization ablation
# ═══════════════════════════════════════════════════════════════════

def mask_merchant_tokens(text: str, merchant: str) -> str:
    """Mask the merchant name at its first contiguous occurrence in the description.

    Span-based (first match only) rather than global token replacement — the
    previous behaviour over-masked whenever the extracted merchant contained a
    generic token like ``card``, ``apple``, ``pay`` or ``mobile``, which
    biased the merchant-memorization ablation.  We try exact substring first,
    then a whitespace-flexible token-sequence match.  When no contiguous span
    is found we return the text unchanged rather than distort non-merchant
    tokens.
    """
    if not merchant or merchant == "unknown_merchant" or not text:
        return text
    # Exact contiguous substring (case-insensitive)
    m = re.search(re.escape(merchant), text, flags=re.I)
    if m:
        return text[: m.start()] + "<merchant>" + text[m.end():]
    # Fallback: token sequence with flexible non-word separators, first match
    toks = [t for t in merchant.split() if t]
    if not toks:
        return text
    pattern = r"\b" + r"\W+".join(re.escape(t) for t in toks) + r"\b"
    m = re.search(pattern, text, flags=re.I)
    if m:
        return text[: m.start()] + "<merchant>" + text[m.end():]
    return text


def merchant_ablation(
    df: pd.DataFrame, seed: int = RANDOM_STATE
) -> Dict[str, Any]:
    """Controlled description-masking ablation on a merchant-disjoint split.

    A. full description        (normalized-basic text)
    B. merchant tokens masked  (<MERCHANT>)
    C. variable ids masked     (structure-aware normalization)
    D. merchant + ids masked
    """
    train_df, test_df, _ = merchant_disjoint_split(df, test_size=0.2, seed=seed)
    label_names = sorted(train_df["category"].unique())
    y_train, y_test = train_df["category"].values, test_df["category"].values

    def _mask(frame):
        out = frame.copy()
        out["_t"] = [
            mask_merchant_tokens(t, m)
            for t, m in zip(out["text_basic"], out["merchant"])
        ]
        return out

    def _mask_both(frame):
        out = frame.copy()
        out["_t"] = [
            mask_merchant_tokens(t, m)
            for t, m in zip(out["text_structured"], out["merchant"])
        ]
        return out

    a_tr, a_te = train_df.assign(_t=train_df["text_basic"]), test_df.assign(_t=test_df["text_basic"])
    b_tr, b_te = _mask(train_df), _mask(test_df)
    c_tr, c_te = train_df.assign(_t=train_df["text_structured"]), test_df.assign(_t=test_df["text_structured"])
    d_tr, d_te = _mask_both(train_df), _mask_both(test_df)

    results: Dict[str, Dict[str, float]] = {}
    for name, (tr, te) in {
        "full": (a_tr, a_te),
        "merchant_masked": (b_tr, b_te),
        "ids_masked": (c_tr, c_te),
        "merchant_ids_masked": (d_tr, d_te),
    }.items():
        wv = fit_word_tfidf(tr["_t"], max_features=15_000)
        Xtr, Xte = vectorize_pair(wv, tr["_t"], te["_t"])
        m = evaluate_model(
            train_logistic_regression(Xtr, y_train, seed=seed),
            Xte, y_test, label_names, f"ablation::{name}",
        )
        results[name] = {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"]}
    return results


# ═══════════════════════════════════════════════════════════════════
#  Format-disjoint generalization
# ═══════════════════════════════════════════════════════════════════

def format_disjoint_eval(
    df: pd.DataFrame,
    holdout_format: str,
    min_examples: int = 500,
    seed: int = RANDOM_STATE,
    merchant_disjoint: bool = True,
) -> Dict[str, Any]:
    """Train on all formats except one; test on the held-out format.

    When ``merchant_disjoint`` is True (default), rows in the holdout format
    whose merchant identity also appears in the training set are dropped from
    the test set.  This isolates *format* generalization from *merchant*
    memorization — without it, the model can succeed on the held-out format
    simply by recognising a merchant it already saw wrapped in a different
    format family.
    """
    fmt_col = df["format"].astype(str)
    test_df = df[fmt_col == holdout_format].reset_index(drop=True)
    train_df = df[fmt_col != holdout_format].reset_index(drop=True)

    excluded_by_merchant = 0
    if merchant_disjoint:
        train_merchants = set(train_df["merchant"])
        mask = ~test_df["merchant"].isin(train_merchants)
        excluded_by_merchant = int((~mask).sum())
        test_df = test_df[mask].reset_index(drop=True)

    if len(test_df) < min_examples:
        return {
            "skipped": True,
            "reason": (
                f"holdout {holdout_format} only {len(test_df)} rows"
                f"{' after merchant-disjoint filter' if merchant_disjoint else ''}"
            ),
            "excluded_by_merchant_overlap": excluded_by_merchant,
        }
    _, m = classify(train_df, test_df, rep="hybrid", model_name="logreg",
                    use_direction=True, seed=seed)
    train_cats = set(train_df["category"])
    test_cats = set(test_df["category"])
    missing = sorted(test_cats - train_cats)
    return {
        "skipped": False,
        "holdout_format": holdout_format,
        "merchant_disjoint": bool(merchant_disjoint),
        "excluded_by_merchant_overlap": excluded_by_merchant,
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "train_categories": len(train_cats),
        "test_categories": len(test_cats),
        "test_categories_absent_from_train": missing,
        "category_support_overlap": round(len(test_cats & train_cats) / max(len(test_cats), 1), 4),
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
    }


# ═══════════════════════════════════════════════════════════════════
#  Calibration
# ═══════════════════════════════════════════════════════════════════

def multiclass_brier(y_int: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score = mean over samples of sum-of-squared-errors
    between the one-hot label and the predicted probability vector.

    ``sklearn.metrics.brier_score_loss`` is binary-only; passing a (N, K)
    probability matrix to it silently discards information.  This helper is
    the standard multi-class generalisation used in the calibration
    literature (Brier 1950; e.g. Guo et al., 2017).
    """
    y_int = np.asarray(y_int, dtype=int)
    probs = np.asarray(probs, dtype=float)
    n, k = probs.shape
    oh = np.zeros_like(probs)
    oh[np.arange(n), y_int] = 1.0
    return float(((probs - oh) ** 2).sum(axis=1).mean())


def _hybrid_representation(
    train_full: pd.DataFrame,
    frames: Sequence[pd.DataFrame],
    representation: str = "hybrid",
    use_direction: bool = True,
    use_structural: bool = True,
    max_features_word: int = 15_000,
    max_features_char: int = 20_000,
) -> List[Any]:
    """Fit vectorizer(s) on ``train_full`` only, transform ``frames`` consistently."""
    from scipy.sparse import hstack

    wv = cv = None
    if representation in ("word", "hybrid"):
        wv = fit_word_tfidf(train_full["description"], max_features=max_features_word)
    if representation in ("char", "hybrid"):
        cv = fit_char_tfidf(train_full["description"], max_features=max_features_char)

    out: List[Any] = []
    for frame in frames:
        parts = []
        if wv is not None:
            parts.append(wv.transform(frame["description"]))
        if cv is not None:
            parts.append(cv.transform(frame["description"]))
        X = parts[0] if len(parts) == 1 else hstack(parts, format="csr")
        out.append(build_feature_matrix(X, frame, use_direction=use_direction,
                                        use_structural=use_structural))
    return out


def calibrate(
    df: pd.DataFrame,
    model_family: str = "linearsvc",
    test_size: float = 0.2,
    cal_size: float = 0.2,
    seed: int = RANDOM_STATE,
    representation: str = "hybrid",
    use_direction: bool = True,
    use_structural: bool = True,
) -> Dict[str, Any]:
    """Fair post-hoc calibration protocol.

    * ``train_full`` / ``test`` — standard 80/20 stratified split (test untouched).
    * The base model is fit on the **full** training split with the same
      hybrid word+char+direction representation used in the main
      Section-10 comparison — the calibration numbers therefore describe the
      model actually reported elsewhere, not a weaker sub-slice model.
    * A held-out calibration slice is carved from the training split; both
      LogReg and LinearSVC are Platt-calibrated on that slice via
      ``CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")``
      (the sklearn-1.6+ replacement for the removed ``cv="prefit"``).
    * Brier is computed as the multi-class Brier score (see
      :func:`multiclass_brier`); the earlier binary ``brier_score_loss`` call
      was ill-defined for a 17-way target.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import log_loss

    train_full, test_df = make_splits(df, test_size=test_size, seed=seed)
    _, cal_fit = make_splits(train_full, test_size=cal_size, seed=seed)

    X_train, X_cal, X_test = _hybrid_representation(
        train_full, [train_full, cal_fit, test_df],
        representation=representation,
        use_direction=use_direction,
        use_structural=use_structural,
    )
    y_train = train_full["category"].values
    y_cal = cal_fit["category"].values
    y_test = test_df["category"].values

    if model_family == "logreg":
        base = train_logistic_regression(X_train, y_train, seed=seed)
    elif model_family == "linearsvc":
        base = train_linear_svc(X_train, y_train, seed=seed)
    else:
        raise ValueError(model_family)

    # sklearn ≥1.6 removed `cv="prefit"`; the replacement is FrozenEstimator.
    calibrated = CalibratedClassifierCV(
        FrozenEstimator(base), method="sigmoid"
    ).fit(X_cal, y_cal)

    probs = calibrated.predict_proba(X_test)
    cal_probs = calibrated.predict_proba(X_cal)

    classes_ = list(calibrated.classes_)
    label_to_int = {c: i for i, c in enumerate(classes_)}
    y_test_int = np.asarray([label_to_int[c] for c in y_test], dtype=int)
    y_cal_int = np.asarray([label_to_int[c] for c in y_cal], dtype=int)
    all_int_labels = list(range(len(classes_)))

    brier = multiclass_brier(y_test_int, probs)
    logloss = float(log_loss(y_test_int, probs, labels=all_int_labels))
    ece = expected_calibration_error(y_test_int, probs, n_bins=10)

    cal_brier = multiclass_brier(y_cal_int, cal_probs)
    cal_logloss = float(log_loss(y_cal_int, cal_probs, labels=all_int_labels))
    cal_ece = expected_calibration_error(y_cal_int, cal_probs, n_bins=10)

    pred = probs.argmax(axis=1)
    acc = float(np.mean(pred == y_test_int))

    # raw (uncalibrated) metrics: LogisticRegression has native probabilities;
    # LinearSVC does not (its raw scores are not calibrated probabilities), so
    # only the calibrated numbers are meaningful for the SVM family.
    if model_family == "logreg":
        raw_probs = base.predict_proba(X_test)
        raw_probs = raw_probs[:, [label_to_int[c] for c in classes_]]  # align class order
        raw_brier = multiclass_brier(y_test_int, raw_probs)
        raw_logloss = float(log_loss(y_test_int, raw_probs, labels=all_int_labels))
        raw_ece = expected_calibration_error(y_test_int, raw_probs, n_bins=10)
    else:
        raw_brier = raw_logloss = raw_ece = None

    return {
        "model_family": model_family,
        "representation": representation,
        "calibration_method": "sigmoid (Platt) via CalibratedClassifierCV(FrozenEstimator(base))",
        "brier_definition": "multi-class Brier (sum of one-hot squared errors, mean over samples)",
        "n_train": int(len(train_full)),
        "n_calibration": int(len(cal_fit)),
        "n_test": int(len(test_df)),
        "accuracy": round(acc, 4),
        "log_loss": round(logloss, 4),
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "raw_log_loss": round(raw_logloss, 4) if raw_logloss is not None else None,
        "raw_brier": round(raw_brier, 4) if raw_brier is not None else None,
        "raw_ece": round(raw_ece, 4) if raw_ece is not None else None,
        "calibration_log_loss": round(cal_logloss, 4),
        "calibration_brier": round(cal_brier, 4),
        "calibration_ece": round(cal_ece, 4),
        "calibration_curve": calibration_curve_data(y_test_int, probs, n_bins=10),
    }


def expected_calibration_error(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """ECE over max-confidence bins (10 bins, equal-width)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if i == 0:
            mask = conf <= hi
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = conf[mask].mean()
        ece += (mask.sum() / len(conf)) * abs(bin_acc - bin_conf)
    return float(ece)


def calibration_curve_data(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> Dict[str, list]:
    """(bin_center, fraction_positives, mean_predicted) for plotting."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, preds = [], [], []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (conf > lo) & (conf <= hi)
        if i == 0:
            mask = conf <= hi
        if mask.sum() == 0:
            continue
        centers.append(float((lo + hi) / 2))
        accs.append(float(correct[mask].mean()))
        preds.append(float(conf[mask].mean()))
    return {"bin_center": centers, "accuracy": accs, "confidence": preds}


# ═══════════════════════════════════════════════════════════════════
#  Selective prediction
# ═══════════════════════════════════════════════════════════════════

def selective_prediction(
    df: pd.DataFrame,
    thresholds: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
    seed: int = RANDOM_STATE,
    representation: str = "hybrid",
    use_direction: bool = True,
    use_structural: bool = True,
    model_family: str = "logreg",
) -> Dict[str, Any]:
    """Confidence-aware abstention on the Platt-calibrated hybrid model.

    Same fair protocol as :func:`calibrate` — base fitted on the full
    training split, calibrated on a held-out slice, test set untouched — so
    the coverage/accuracy curve describes the deployment-grade model rather
    than a weaker sub-slice model.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import accuracy_score, f1_score

    train_full, test_df = make_splits(df, test_size=0.2, seed=seed)
    _, cal_fit = make_splits(train_full, test_size=0.2, seed=seed)

    X_train, X_cal, X_test = _hybrid_representation(
        train_full, [train_full, cal_fit, test_df],
        representation=representation,
        use_direction=use_direction,
        use_structural=use_structural,
    )
    y_train = train_full["category"].values
    y_cal = cal_fit["category"].values
    y_test = test_df["category"].values

    if model_family == "logreg":
        base = train_logistic_regression(X_train, y_train, seed=seed)
    elif model_family == "linearsvc":
        base = train_linear_svc(X_train, y_train, seed=seed)
    else:
        raise ValueError(model_family)

    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(X_cal, y_cal)

    probs = cal.predict_proba(X_test)
    classes_ = list(cal.classes_)
    label_to_int = {c: i for i, c in enumerate(classes_)}
    y_int = np.asarray([label_to_int[c] for c in y_test], dtype=int)

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)

    rows: Dict[str, Any] = {}
    for t in thresholds:
        accept = conf >= t
        n_acc = int(accept.sum())
        coverage = float(n_acc / len(conf))
        if n_acc == 0:
            rows[str(t)] = {
                "threshold": t, "coverage": 0.0, "accuracy": None, "macro_f1": None,
                "abstention_rate": 1.0, "accepted_count": 0,
            }
            continue
        y_acc = y_int[accept]
        p_acc = pred[accept]
        rows[str(t)] = {
            "threshold": t,
            "coverage": round(coverage, 4),
            "abstention_rate": round(1.0 - coverage, 4),
            "accuracy": round(float(accuracy_score(y_acc, p_acc)), 4),
            "macro_f1": round(float(f1_score(y_acc, p_acc, average="macro", zero_division=0)), 4),
            "accepted_count": n_acc,
        }
    return {
        "rows": rows,
        "model_family": model_family,
        "representation": representation,
        "calibration_method": "sigmoid (Platt) via CalibratedClassifierCV(FrozenEstimator(base))",
    }


# ═══════════════════════════════════════════════════════════════════
#  Learning curve
# ═══════════════════════════════════════════════════════════════════

def learning_curve(
    df: pd.DataFrame,
    train_sizes: Sequence[int] = (5_000, 10_000, 20_000, 40_000),
    seed: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Fixed test set; train sizes sampled from the training split."""
    _, test_df = make_splits(df, test_size=0.2, seed=seed)
    train_full, _ = make_splits(df, test_size=0.2, seed=seed)
    label_names = sorted(train_full["category"].unique())
    y_test = test_df["category"].values

    rows = []
    for size in train_sizes:
        n = min(size, len(train_full))
        train_df = train_full.sample(n=n, random_state=seed)
        wv = fit_word_tfidf(train_df["description"], max_features=15_000)
        Xtr, Xte = vectorize_pair(wv, train_df["description"], test_df["description"])
        Xtr = build_feature_matrix(Xtr, train_df, use_direction=True)
        Xte = build_feature_matrix(Xte, test_df, use_direction=True)
        m = evaluate_model(
            train_logistic_regression(Xtr, train_df["category"].values, seed=seed),
            Xte, y_test, label_names, f"learning_curve::{n}",
        )
        rows.append({"train_size": int(n), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"]})
    return {"rows": rows}


# ═══════════════════════════════════════════════════════════════════
#  Error analysis
# ═══════════════════════════════════════════════════════════════════

def error_analysis(
    df: pd.DataFrame,
    model: Any = None,
    seed: int = RANDOM_STATE,
    prefit: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Representative misclassifications + worst categories.

    Preferred usage — supply ``prefit`` so the analysis reflects the exact
    model reported elsewhere (no silent refit)::

        error_analysis(df, prefit={
            "model":     best_model,
            "X_test":    Xte_full,        # already featurised
            "test_df":   test_df,
            "labels":    sorted(train_df["category"].unique()),
        })

    Legacy fallback — retained for backward compatibility — fits a fresh
    word-only LogReg on the standard 80/20 split and *emits a warning* that
    the reported errors do not correspond to the main comparison model.
    """
    from sklearn.metrics import classification_report

    if prefit is not None:
        fitted = prefit["model"]
        X_te = prefit["X_test"]
        test_df = prefit["test_df"].reset_index(drop=True)
        y_true = np.asarray(test_df["category"].values)
        label_names = list(prefit.get("labels") or sorted(test_df["category"].unique()))
    else:
        logger.warning(
            "error_analysis: no `prefit` supplied — fitting a fresh word-only "
            "LogReg; reported errors will NOT reflect the main comparison model."
        )
        train_df, test_df = make_splits(df, test_size=0.2, seed=seed)
        wv = fit_word_tfidf(train_df["description"], max_features=15_000)
        Xtr, Xte = vectorize_pair(wv, train_df["description"], test_df["description"])
        Xtr = build_feature_matrix(Xtr, train_df, use_direction=True)
        Xte = build_feature_matrix(Xte, test_df, use_direction=True)
        if model is None:
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(max_iter=8000, solver="lbfgs", random_state=seed)
        fitted = model.fit(Xtr, train_df["category"].values)
        X_te = Xte
        y_true = test_df["category"].values
        label_names = sorted(train_df["category"].unique())

    y_pred = fitted.predict(X_te)
    try:
        probs = fitted.predict_proba(X_te)
        conf = probs.max(axis=1)
    except AttributeError:
        conf = np.full(len(y_true), np.nan)

    mis = test_df.copy()
    mis["true"] = y_true
    mis["predicted"] = y_pred
    mis["confidence"] = conf
    mis = mis[mis["true"] != mis["predicted"]]

    rep = classification_report(
        y_true, y_pred, target_names=label_names, labels=label_names,
        output_dict=True, zero_division=0,
    )
    per_class = {
        k: {"precision": v["precision"], "recall": v["recall"],
            "f1": v["f1-score"], "support": v["support"]}
        for k, v in rep.items() if k not in ("accuracy", "macro avg", "weighted avg")
    }
    worst = sorted(per_class.items(), key=lambda kv: kv[1]["f1"])[:5]
    best = sorted(per_class.items(), key=lambda kv: kv[1]["f1"], reverse=True)[:5]

    examples = []
    if len(mis):
        for _, row in mis.sample(min(25, len(mis)), random_state=seed).iterrows():
            c = row["confidence"]
            is_num = isinstance(c, (int, float, np.floating))
            examples.append({
                "true": str(row["true"]),
                "predicted": str(row["predicted"]),
                "description": str(row["description"]),
                "confidence": round(float(c), 4) if is_num and not np.isnan(c) else None,
            })
    return {
        "n_misclassified": int(len(mis)),
        "misclassification_rate": round(float(len(mis)) / len(test_df), 4),
        "worst_categories": {k: v for k, v in worst},
        "best_categories": {k: v for k, v in best},
        "examples": examples,
    }


# ═══════════════════════════════════════════════════════════════════
#  Artifact helpers
# ═══════════════════════════════════════════════════════════════════

def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════
#  Strict generalization: merchant-disjoint + exact-description-disjoint
# ═══════════════════════════════════════════════════════════════════

def strict_disjoint_split(
    df: pd.DataFrame, test_size: float = 0.2, seed: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Merchant-disjoint AND exact-description-disjoint split.

    Construction: start from the merchant-disjoint split, then drop TRAIN rows
    whose exact description also appears in TEST (keeping the full test set as
    the evaluation support).  The resulting pair has zero canonical-merchant
    overlap and zero exact-description overlap; the train reduction is
    reported explicitly rather than silently discarding data.
    """
    g_train, g_test, info = merchant_disjoint_split(df, test_size=test_size, seed=seed)
    test_descs = set(g_test["description"])
    drop = g_train["description"].isin(test_descs)
    n_dropped = int(drop.sum())
    s_train = g_train[~drop].reset_index(drop=True)
    info.update({
        "exact_desc_overlap": int(len(set(s_train["description"]) & test_descs)),
        "train_rows_dropped_for_desc_overlap": n_dropped,
        "train_size_strict": int(len(s_train)),
        "test_size_strict": int(len(g_test)),
        "train_share_retained": round(len(s_train) / len(g_train), 4),
    })
    assert info["exact_desc_overlap"] == 0, "exact description overlap must be 0"
    assert info["merchant_overlap"] == 0, "canonical merchant overlap must be 0"
    return s_train, g_test, info


# ═══════════════════════════════════════════════════════════════════
#  Duplicate-description audit
# ═══════════════════════════════════════════════════════════════════

def duplicate_audit(
    df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Duplicate-description structure (duplicates are NOT removed globally).

    Reports exact-duplicate descriptions, duplicates spanning multiple
    categories / merchants, and — when the random-split frames are supplied —
    the exact-description overlap between random-split train and test (which
    is why the random split can look unusually easy).
    """
    vc = df["description"].value_counts()
    dup_descs = vc[vc > 1]
    cat_n = df.groupby("description")["category"].nunique()
    merch_n = df.groupby("description")["merchant"].nunique()
    multi_cat = int((cat_n.reindex(dup_descs.index) > 1).sum()) if len(dup_descs) else 0
    multi_merch = int((merch_n.reindex(dup_descs.index) > 1).sum()) if len(dup_descs) else 0
    out: Dict[str, Any] = {
        "n_unique_descriptions": int(vc.shape[0]),
        "n_duplicate_descriptions": int(len(dup_descs)),
        "n_duplicate_rows": int(df.duplicated(subset=["description"]).sum()),
        "n_duplicate_full_rows": int(df.duplicated().sum()),
        "max_occurrences_of_one_description": int(vc.max()),
        "dup_descs_spanning_multiple_categories": multi_cat,
        "dup_descs_spanning_multiple_merchants": multi_merch,
    }
    if train_df is not None and test_df is not None:
        tr_s, te_s = set(train_df["description"]), set(test_df["description"])
        out["random_split_desc_overlap"] = int(len(tr_s & te_s))
        out["random_split_test_desc_share_seen_in_train"] = round(len(tr_s & te_s) / max(len(te_s), 1), 4)
    return out


# ═══════════════════════════════════════════════════════════════════
#  Merchant canonicalization audit
# ═══════════════════════════════════════════════════════════════════

def canonicalization_audit(
    df: pd.DataFrame, n_merchants: int = 8, n_examples: int = 3
) -> Dict[str, Any]:
    """Evidence for the merchant canonicalization rules.

    Shows (a) alias groups — distinct raw descriptions mapping to the same
    canonical merchant — and (b) rule evidence: representative raw
    descriptions per stripping rule (store numbers, order IDs, provider
    prefixes, US-address blocks, ACH reference IDs) with the canonical
    merchant produced.  No semantic mappings are invented: every example is
    drawn from the actual data.
    """
    canon_count = int(df["merchant"].nunique())
    g = df.groupby("merchant")["description"].nunique()
    alias_merchants = g[g > 1].sort_values(ascending=False)
    groups = []
    for m in alias_merchants.head(n_merchants).index:
        raws = df.loc[df["merchant"] == m, "description"].drop_duplicates().head(n_examples)
        groups.append({
            "canonical": str(m),
            "n_raw_forms": int(g[m]),
            "raw_examples": [str(r) for r in raws],
        })
    rule_subsets = {
        "store number (#123)": df[df["description"].str.contains(r"#\d+", regex=True, na=False)],
        "order id (*ABC)": df[df["description"].str.contains(r"\*\w+", regex=True, na=False)],
        "provider prefix (SQ*/PP*/PYPL*/TST*/CLV*)": df[df["description"].str.contains(
            r"(?i)^(?:sq|pp|pypl|tst|clv)\s*\*", regex=True, na=False)],
        "US address block (ZIP ST USA)": df[df["description"].str.contains(
            r"(?i)\b\d{5}(?:-\d{4})?\s*[a-z]{2}\s*usa\b", regex=True, na=False)],
        "ACH reference (PPD/WEB ID)": df[df["description"].str.contains(
            r"(?i)\b(?:ppd|web)\s*id\b", regex=True, na=False)],
    }
    rules: Dict[str, list] = {}
    for rule, sub in rule_subsets.items():
        rules[rule] = [
            {"raw": str(d), "canonical_merchant": str(m)}
            for d, m in zip(sub["description"].head(2), sub["merchant"].head(2))
        ]
    return {
        "n_unique_raw_descriptions": int(df["description"].nunique()),
        "n_unique_canonical_merchants": canon_count,
        "n_merchant_alias_groups": int(len(alias_merchants)),
        "top_alias_groups": groups,
        "rule_evidence": rules,
    }


# ═══════════════════════════════════════════════════════════════════
#  Selective prediction with validation-based threshold selection
# ═══════════════════════════════════════════════════════════════════

def select_abstention_threshold(
    df: pd.DataFrame,
    coverage_target: float = 0.90,
    thresholds: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
    seed: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Threshold selected on the CALIBRATION slice; frozen threshold tested once.

    Selection objective: max Macro-F1 subject to coverage >= ``coverage_target``
    evaluated on the held-out calibration slice.  The frozen threshold is then
    evaluated ONCE on the test set (no test leakage in the choice).
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.metrics import f1_score

    train_full, test_df = make_splits(df, test_size=0.2, seed=seed)
    _, cal_fit = make_splits(train_full, test_size=0.2, seed=seed)
    X_train, X_cal, X_test = _hybrid_representation(
        train_full, [train_full, cal_fit, test_df])
    y_train = train_full["category"].values
    y_cal = cal_fit["category"].values
    y_test = test_df["category"].values

    base = train_logistic_regression(X_train, y_train, seed=seed)
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(X_cal, y_cal)

    classes_ = list(cal.classes_)
    label_to_int = {c: i for i, c in enumerate(classes_)}
    y_cal_int = np.asarray([label_to_int[c] for c in y_cal], dtype=int)
    y_test_int = np.asarray([label_to_int[c] for c in y_test], dtype=int)

    def _eval(probs: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, Any]]:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        rows: Dict[str, Dict[str, Any]] = {}
        for t in thresholds:
            acc = conf >= t
            if acc.sum() == 0:
                rows[str(t)] = {"coverage": 0.0, "macro_f1": None, "accuracy": None}
                continue
            rows[str(t)] = {
                "coverage": round(float(acc.mean()), 4),
                "macro_f1": round(float(f1_score(y[acc], pred[acc], average="macro",
                                                 zero_division=0)), 4),
                "accuracy": round(float((pred[acc] == y[acc]).mean()), 4),
            }
        return rows

    cal_rows = _eval(cal.predict_proba(X_cal), y_cal_int)
    eligible = {t: r for t, r in cal_rows.items()
                if r["coverage"] >= coverage_target and r["macro_f1"] is not None}
    if not eligible:
        best_t = str(thresholds[0])
        reason = (f"no threshold reached coverage >= {coverage_target} on the calibration "
                  f"slice; fell back to the lowest threshold (documented, not tuned on test)")
    else:
        best_t = max(eligible, key=lambda t: eligible[t]["macro_f1"])
        reason = f"max Macro-F1 subject to coverage >= {coverage_target} (calibration slice)"
    test_rows = _eval(cal.predict_proba(X_test), y_test_int)
    return {
        "selected_threshold": float(best_t),
        "selection_objective": reason,
        "coverage_target": coverage_target,
        "calibration_slice_table": cal_rows,
        "frozen_threshold_test": test_rows[best_t],
        "test_full_table": test_rows,
    }


# ═══════════════════════════════════════════════════════════════════
#  Final calibrated pipeline (deployment model)
# ═══════════════════════════════════════════════════════════════════

def final_calibrated_model(
    df: pd.DataFrame,
    model_family: str = "linearsvc",
    use_structural: bool = False,
    seed: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """The FINAL deployment pipeline: base classifier -> Platt calibration.

    Representation is the ablation winner (§9.1): word+char TF-IDF + debit/
    credit direction; structural statistics are omitted because they provided
    no improvement (and slightly reduced Macro-F1 in the ablation).  The
    calibration slice is carved from the training split; the test set is
    untouched.  Returns the calibrated model together with its test features,
    labels and calibrated probabilities so every downstream confidence-based
    analysis uses the SAME model.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator

    train_full, test_df = make_splits(df, test_size=0.2, seed=seed)
    _, cal_fit = make_splits(train_full, test_size=0.2, seed=seed)
    X_train, X_cal, X_test = _hybrid_representation(
        train_full, [train_full, cal_fit, test_df],
        use_direction=True, use_structural=use_structural)
    y_train = train_full["category"].values
    y_cal = cal_fit["category"].values
    y_test = test_df["category"].values

    if model_family == "linearsvc":
        base = train_linear_svc(X_train, y_train, seed=seed)
    elif model_family == "logreg":
        base = train_logistic_regression(X_train, y_train, seed=seed)
    else:
        raise ValueError(model_family)

    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(X_cal, y_cal)
    classes_ = list(cal.classes_)
    label_to_int = {c: i for i, c in enumerate(classes_)}
    y_cal_int = np.asarray([label_to_int[c] for c in y_cal], dtype=int)
    y_test_int = np.asarray([label_to_int[c] for c in y_test], dtype=int)
    return {
        "model": cal,
        "X_test": X_test,
        "test_df": test_df.reset_index(drop=True),
        "labels": sorted(train_full["category"].unique()),
        "model_family": model_family,
        "representation": "word+char+direction" + ("+structural" if use_structural else ""),
        "n_train": int(len(train_full)),
        "n_calibration": int(len(cal_fit)),
        "n_test": int(len(test_df)),
        "cal_proba": cal.predict_proba(X_cal),
        "test_proba": cal.predict_proba(X_test),
        "y_cal_int": y_cal_int,
        "y_test_int": y_test_int,
        "classes_": classes_,
    }


def abstention_table(
    proba: np.ndarray, y_int: np.ndarray,
    thresholds: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
) -> Dict[str, Dict[str, Any]]:
    """Coverage/accuracy/Macro-F1 per confidence threshold (pure function)."""
    from sklearn.metrics import accuracy_score, f1_score

    conf = np.asarray(proba).max(axis=1)
    pred = np.asarray(proba).argmax(axis=1)
    y_int = np.asarray(y_int, dtype=int)
    rows: Dict[str, Dict[str, Any]] = {}
    for t in thresholds:
        accept = conf >= t
        n_acc = int(accept.sum())
        if n_acc == 0:
            rows[str(t)] = {"threshold": t, "coverage": 0.0, "abstention_rate": 1.0,
                            "accuracy": None, "macro_f1": None, "accepted_count": 0}
            continue
        rows[str(t)] = {
            "threshold": t,
            "coverage": round(float(accept.mean()), 4),
            "abstention_rate": round(1.0 - float(accept.mean()), 4),
            "accuracy": round(float(accuracy_score(y_int[accept], pred[accept])), 4),
            "macro_f1": round(float(f1_score(y_int[accept], pred[accept], average="macro",
                                             zero_division=0)), 4),
            "accepted_count": n_acc,
        }
    return rows


def select_threshold_from_probas(
    cal_proba: np.ndarray, y_cal_int: np.ndarray,
    test_proba: np.ndarray, y_test_int: np.ndarray,
    coverage_target: float = 0.90,
    thresholds: Sequence[float] = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99),
) -> Dict[str, Any]:
    """Select the abstention threshold on the CALIBRATION slice, test frozen.

    Objective: max Macro-F1 subject to coverage >= ``coverage_target``,
    evaluated on the calibration probabilities (never on test).  The frozen
    threshold is then evaluated once on the test probabilities.
    """
    cal_rows = abstention_table(cal_proba, y_cal_int, thresholds)
    test_rows = abstention_table(test_proba, y_test_int, thresholds)
    eligible = {t: r for t, r in cal_rows.items()
                if r["coverage"] >= coverage_target and r["macro_f1"] is not None}
    if not eligible:
        best_t = str(thresholds[0])
        reason = (f"no threshold reached coverage >= {coverage_target} on the calibration "
                  f"slice; fell back to the lowest threshold (documented, not tuned on test)")
    else:
        best_t = max(eligible, key=lambda t: eligible[t]["macro_f1"])
        reason = f"max Macro-F1 subject to coverage >= {coverage_target} (calibration slice)"
    return {
        "selected_threshold": float(best_t),
        "selection_objective": reason,
        "coverage_target": coverage_target,
        "calibration_slice_table": cal_rows,
        "frozen_threshold_test": test_rows[best_t],
        "test_full_table": test_rows,
    }


# ═══════════════════════════════════════════════════════════════════
#  Final merchant-disjoint calibration pipeline (PRIMARY regime)
# ═══════════════════════════════════════════════════════════════════

def final_merchant_disjoint_pipeline(
    df: pd.DataFrame,
    test_size: float = 0.2,
    cal_size: float = 0.2,
    seed: int = RANDOM_STATE,
) -> Dict[str, Any]:
    """Final deployment pipeline aligned with the PRIMARY merchant-disjoint regime.

    Partitions (protocol):

    | partition | role |
    |---|---|
    | g_train / g_test | merchant-disjoint split (canonical overlap = 0) |
    | g_fit / g_cal | stratified 80/20 split of g_train |
    | g_fit | TF-IDF fitting + LinearSVC training |
    | g_cal | sigmoid probability calibration + confidence-threshold selection |
    | g_test | final untouched evaluation (never used for fitting) |

    Merchant overlap between g_fit and g_cal is permitted (forcing it would
    only shrink training data); the hard requirement is **no test leakage**:
    g_test is untouched by vectorizers, model, calibration and threshold
    selection.  The representation is the ablation winner
    (word+char+direction; structural statistics omitted).  The confidence
    threshold is selected on g_cal by the predefined rule (max Macro-F1
    subject to coverage >= 90%; ties broken toward the lowest threshold).
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.frozen import FrozenEstimator
    from sklearn.model_selection import train_test_split

    g_train, g_test, split_info = merchant_disjoint_split(df, test_size=test_size, seed=seed)
    g_fit, g_cal = train_test_split(g_train, test_size=cal_size,
                                    stratify=g_train["category"], random_state=seed)
    g_fit = g_fit.reset_index(drop=True)
    g_cal = g_cal.reset_index(drop=True)

    # vectorizers fitted on g_fit ONLY; g_cal/g_test transformed with frozen vectorizers
    wv = fit_word_tfidf(g_fit["description"], max_features=15_000)
    cv = fit_char_tfidf(g_fit["description"], max_features=20_000)

    def _transform(frame: pd.DataFrame) -> Any:
        from scipy.sparse import hstack
        X = hstack([wv.transform(frame["description"]),
                    cv.transform(frame["description"])], format="csr")
        return build_feature_matrix(X, frame, use_direction=True, use_structural=False)

    X_fit, X_cal, X_test = _transform(g_fit), _transform(g_cal), _transform(g_test)

    # LinearSVC on g_fit with the already-selected (frozen) hyperparameters
    base = train_linear_svc(X_fit, g_fit["category"].values, seed=seed)
    # sigmoid calibration on g_cal ONLY — never g_test
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(
        X_cal, g_cal["category"].values)

    classes_ = list(cal.classes_)
    label_to_int = {c: i for i, c in enumerate(classes_)}
    y_cal_int = np.asarray([label_to_int[c] for c in g_cal["category"].values], dtype=int)
    y_test_int = np.asarray([label_to_int[c] for c in g_test["category"].values], dtype=int)
    cal_proba = cal.predict_proba(X_cal)
    test_proba = cal.predict_proba(X_test)

    sel = select_threshold_from_probas(
        cal_proba, y_cal_int, test_proba, y_test_int, coverage_target=0.90)

    m_fit, m_cal, m_test = set(g_fit["merchant"]), set(g_cal["merchant"]), set(g_test["merchant"])
    return {
        "model": cal,
        "transform": _transform,          # frozen vectorizers + feature builder
        "g_fit": g_fit, "g_cal": g_cal, "g_test": g_test,
        "X_fit": X_fit, "X_cal": X_cal, "X_test": X_test,
        "y_cal_int": y_cal_int, "y_test_int": y_test_int,
        "cal_proba": cal_proba, "test_proba": test_proba,
        "labels": sorted(g_fit["category"].unique()),
        "classes_": classes_,
        "threshold_selection": sel,
        "merchant_overlap": {
            "g_fit_x_g_cal": int(len(m_fit & m_cal)),
            "g_fit_x_g_test": int(len(m_fit & m_test)),
            "g_cal_x_g_test": int(len(m_cal & m_test)),
            "canonical_g_train_x_g_test": int(split_info["merchant_overlap"]),
        },
        "split_sizes": {"g_train": int(len(g_train)), "g_fit": int(len(g_fit)),
                        "g_cal": int(len(g_cal)), "g_test": int(len(g_test))},
        "representation": "word+char+direction (ablation winner; structural omitted)",
        "classifier": "LinearSVC",
        "calibration_method": "sigmoid (Platt) via CalibratedClassifierCV(FrozenEstimator(base))",
        "random_state": int(seed),
    }


def strict_frozen_evaluation(
    df: pd.DataFrame, pipeline: Dict[str, Any], seed: int = RANDOM_STATE
) -> Dict[str, Any]:
    """Apply the FROZEN calibrated model + threshold to the strict test set.

    The strict benchmark (merchant + exact-description disjoint) is a
    held-out robustness test: nothing is retuned, the vectorizers, model,
    calibration and threshold are exactly the final merchant-disjoint
    pipeline's.
    """
    from sklearn.metrics import accuracy_score, f1_score, log_loss

    s_train, s_test, info = strict_disjoint_split(df, test_size=0.2, seed=seed)
    X_test = pipeline["transform"](s_test)
    model = pipeline["model"]
    proba = model.predict_proba(X_test)
    classes_ = list(model.classes_)
    label_to_int = {c: i for i, c in enumerate(classes_)}
    y_int = np.asarray([label_to_int[c] for c in s_test["category"].values], dtype=int)
    pred = proba.argmax(axis=1)
    labels_all = list(range(len(classes_)))

    t = pipeline["threshold_selection"]["selected_threshold"]
    conf = proba.max(axis=1)
    accept = conf >= t

    return {
        "merchant_overlap": int(info["merchant_overlap"]),
        "exact_desc_overlap": int(info["exact_desc_overlap"]),
        "train_size_strict": int(info["train_size_strict"]),
        "test_size_strict": int(info["test_size_strict"]),
        "accuracy": round(float(accuracy_score(y_int, pred)), 4),
        "macro_f1": round(float(f1_score(y_int, pred, average="macro", zero_division=0,
                                          labels=labels_all)), 4),
        "weighted_f1": round(float(f1_score(y_int, pred, average="weighted", zero_division=0,
                                            labels=labels_all)), 4),
        "log_loss": round(float(log_loss(y_int, proba, labels=labels_all)), 4),
        "brier": round(multiclass_brier(y_int, proba), 4),
        "ece": round(expected_calibration_error(y_int, proba, n_bins=10), 4),
        "selective": {
            "frozen_threshold": float(t),
            "coverage": round(float(accept.mean()), 4),
            "abstention_rate": round(1.0 - float(accept.mean()), 4),
            "accuracy_accepted": round(float(accuracy_score(y_int[accept], pred[accept])), 4)
            if accept.any() else None,
            "macro_f1_accepted": round(float(f1_score(y_int[accept], pred[accept],
                                                      average="macro", zero_division=0,
                                                      labels=labels_all)), 4)
            if accept.any() else None,
        },
        "note": "strict generalization evaluation — frozen pipeline, no retuning",
    }


# ═══════════════════════════════════════════════════════════════════
#  Final comparison: 3 classifiers x {random, merchant-disjoint, strict}
# ═══════════════════════════════════════════════════════════════════

def final_comparison(
    df: pd.DataFrame, seed: int = RANDOM_STATE
) -> Dict[str, Dict[str, float]]:
    """Macro-F1 / accuracy / weighted-F1 of each classifier under three regimes.

    Same representation (hybrid word+char+direction — the §9.1 ablation
    winner; structural statistics are omitted as they did not improve
    performance), same hyperparameters, same labels — only the split regime
    changes.  The merchant-disjoint and strict splits are the generalization
    benchmarks; the random split is the in-distribution reference.
    """
    random_train, random_test = make_splits(df, test_size=0.2, seed=seed)
    g_train, g_test, _ = merchant_disjoint_split(df, test_size=0.2, seed=seed)
    s_train, s_test, _ = strict_disjoint_split(df, test_size=0.2, seed=seed)
    out: Dict[str, Dict[str, float]] = {}
    for model in ("logreg", "linearsvc", "xgboost"):
        _, mr = classify(random_train, random_test, rep="hybrid", model_name=model,
                         use_direction=True, use_structural=False, seed=seed)
        _, mg = classify(g_train, g_test, rep="hybrid", model_name=model,
                         use_direction=True, use_structural=False, seed=seed)
        _, ms = classify(s_train, s_test, rep="hybrid", model_name=model,
                         use_direction=True, use_structural=False, seed=seed)
        out[model] = {
            "random_macro_f1": mr["macro_f1"], "random_accuracy": mr["accuracy"],
            "random_weighted_f1": mr["weighted_f1"],
            "merchant_disjoint_macro_f1": mg["macro_f1"], "merchant_disjoint_accuracy": mg["accuracy"],
            "merchant_disjoint_weighted_f1": mg["weighted_f1"],
            "strict_macro_f1": ms["macro_f1"], "strict_accuracy": ms["accuracy"],
            "strict_weighted_f1": ms["weighted_f1"],
        }
        logger.info("final comparison %s: random %.4f | merchant-disjoint %.4f | strict %.4f",
                    model, mr["macro_f1"], mg["macro_f1"], ms["macro_f1"])
    return out


# ═══════════════════════════════════════════════════════════════════
#  Main CLI (regenerates everything)
# ═══════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Transactions training + evaluation pipeline")
    parser.add_argument("--out", type=str, default="./out")
    parser.add_argument("--reports", type=str, default="./reports")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Optional cap for fast smoke runs")
    args = parser.parse_args()

    seed_everything(RANDOM_STATE)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir = Path(args.reports)
    rep_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw_dataframe(DATASET_ID, max_rows=args.max_rows)

    # ── ETL / feature extraction ───────────────────────────────────
    df["direction"] = df["description"].apply(extract_direction)
    df["merchant"] = df["description"].apply(extract_merchant)
    df["format"] = df["description"].apply(detect_format)
    df["city"] = df["description"].apply(extract_city)
    df["text_basic"] = df["description"].apply(normalize_basic)
    df["text_structured"] = df["description"].apply(normalize_structured)

    schema = inspect_schema(df)
    write_json({"dataset": DATASET_ID, "schema": schema}, out_dir / "dataset_summary.json")
    logger.info("Saved dataset_summary.json")

    # ── Class distribution / duplicates / direction / format ──────
    summary = {
        "rows": int(len(df)),
        "categories": int(df["category"].nunique()),
        "merchants": int(df["merchant"].nunique()),
        "cities": int(df["city"].nunique()),
        "formats": format_counts(df),
        "direction": dict(Counter(df["direction"])),
        "duplicate_descriptions": int(df["description"].duplicated().sum()),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "class_distribution": {k: int(v) for k, v in df["category"].value_counts().items()},
    }
    write_json(summary, out_dir / "dataset_summary.json")
    logger.info("Dataset summary written")

    # ── Random split baseline (word / char / hybrid / full) ───────
    train_df, test_df = make_splits(df, test_size=0.2, seed=RANDOM_STATE)

    abl = feature_ablation(train_df, test_df, seed=RANDOM_STATE)
    write_json(abl, out_dir / "feature_ablation.json")

    model_results = {}
    for model_name in ("logreg", "linearsvc", "xgboost"):
        _, m = classify(train_df, test_df, rep="hybrid", model_name=model_name,
                        use_direction=True, use_structural=True, seed=RANDOM_STATE)
        model_results[model_name] = m
    write_json(model_results, out_dir / "classification_metrics.json")

    # ── Merchant-disjoint generalization ───────────────────────────
    g_train, g_test, split_info = merchant_disjoint_split(df, test_size=0.2, seed=RANDOM_STATE)
    gen = {}
    for model_name in ("logreg", "linearsvc", "xgboost"):
        _, m = classify(g_train, g_test, rep="hybrid", model_name=model_name,
                        use_direction=True, seed=RANDOM_STATE)
        gen[model_name] = m
    gen["merchant_overlap_fraction_random"] = round(
        merchant_overlap_fraction(train_df, test_df), 4
    )
    gen["split"] = split_info
    write_json(gen, out_dir / "generalization_metrics.json")

    # ── Merchant memorization ablation ─────────────────────────────
    mem_abl = merchant_ablation(df, seed=RANDOM_STATE)
    write_json(mem_abl, out_dir / "merchant_ablation.json")

    # ── Format generalization ──────────────────────────────────────
    fmt_eval = {}
    for fmt in sorted(df["format"].astype(str).unique()):
        fmt_eval[fmt] = format_disjoint_eval(df, fmt, min_examples=500, seed=RANDOM_STATE)
    write_json(fmt_eval, out_dir / "format_generalization.json")

    # ── Calibration ────────────────────────────────────────────────
    cal_lr = calibrate(df, model_family="logreg", seed=RANDOM_STATE)
    cal_svc = calibrate(df, model_family="linearsvc", seed=RANDOM_STATE)
    write_json({"logreg": cal_lr, "linearsvc": cal_svc}, out_dir / "calibration_metrics.json")

    # ── Selective prediction ───────────────────────────────────────
    sel = selective_prediction(df, seed=RANDOM_STATE)
    write_json(sel, out_dir / "selective_prediction.json")

    # ── Learning curve ─────────────────────────────────────────────
    lc = learning_curve(df, seed=RANDOM_STATE)
    write_json(lc, out_dir / "learning_curve.json")

    # ── Error analysis ─────────────────────────────────────────────
    from sklearn.linear_model import LogisticRegression
    er = error_analysis(df, LogisticRegression(max_iter=8000, solver="lbfgs", random_state=RANDOM_STATE), seed=RANDOM_STATE)
    write_json(er, out_dir / "error_analysis.json")

    logger.info("All artifacts written to %s", out_dir)


if __name__ == "__main__":
    main()
