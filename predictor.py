"""
Transactions standalone inference core (no Streamlit dependency).

Loads artifacts/transactions_linear_bundle.joblib produced by train_export.py and exposes
a single entry point: TransactionsPredictor.predict(texts, model) -> dict

Standalone: depends only on the bundle file next to this project (numpy/pandas/
scikit-learn/scipy/joblib). No HuggingFace download, no notebook state.
"""
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import binarize
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BUNDLE = os.path.join(HERE, "artifacts", "transactions_linear_bundle.joblib")
DIR_COLS = ["debit", "credit", "unknown"]

MODEL_INFO = {
    "bm25": {
        "name": "BM25 + LinearSVC",
        "paper": "Robertson & Zaragoza (2009), The Probabilistic Relevance Framework: BM25 and Beyond",
        "role": "Primary estimator",
    },
    "nbsvm": {
        "name": "NBSVM (NB log-ratios)",
        "paper": "Wang & Manning (2012), Baselines and Bigrams (ACL)",
        "role": "Secondary estimator",
    },
    "linearsvm": {
        "name": "Linear SVM (TF-IDF)",
        "paper": "Original notebook baseline",
        "role": "Baseline reference",
    },
}


def _direction_of(desc: str) -> str:
    d = str(desc).strip().lower()
    return "debit" if d.startswith("[debit]") else ("credit" if d.startswith("[credit]") else "unknown")


def _direction_block(n, directions):
    d = pd.get_dummies(pd.Series(list(directions))).reindex(columns=DIR_COLS, fill_value=0)
    assert d.shape[0] == n
    return csr_matrix(d.values.astype(np.float64))


def _softmax(a):
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _bm25_transform(cv, texts, idf, avg_len, k1, b):
    C = cv.transform(texts).tocoo()
    if C.nnz == 0:
        return sp.csr_matrix((len(texts), len(idf)), dtype=np.float64)
    L = np.asarray(C.sum(axis=1)).ravel()
    denom = C.data + k1 * (1 - b + b * L[C.row] / max(avg_len, 1e-9))
    vals = idf[C.col] * (C.data * (k1 + 1)) / denom
    return sp.csr_matrix((vals, (C.row, C.col)), shape=C.shape)


class TransactionsPredictor:
    def __init__(self, bundle_path: str = DEFAULT_BUNDLE):
        if not os.path.exists(bundle_path):
            raise FileNotFoundError(
                f"Serving bundle not found: {bundle_path}\n"
                f"Generate it first with:  python train_export.py"
            )
        self.bundle = joblib.load(bundle_path)
        self.labels = self.bundle["labels"]
        self.models = self.bundle["models"]
        self.k1 = float(self.bundle.get("k1", 1.5))
        self.b = float(self.bundle.get("b", 0.75))

    # ---------- public API ----------
    def available_models(self):
        return list(self.models.keys())

    def model_card(self, model_key: str) -> dict:
        info = dict(MODEL_INFO.get(model_key, {"name": model_key}))
        info["macro_f1"] = self.bundle.get("metrics", {}).get(info.get("name", ""), {}).get("macro_f1")
        return info

    def predict(self, texts, model: str = "bm25") -> dict:
        """texts: list[str]. Returns dict(labels, pred, score_df)."""
        if isinstance(texts, str):
            texts = [texts]
        texts = [str(t) for t in texts]
        if not texts:
            raise ValueError("No input texts provided.")
        if model not in self.models:
            raise KeyError(f"Unknown model '{model}'. Available: {self.available_models()}")

        dirs = [_direction_of(t) for t in texts]
        D = _direction_block(len(texts), dirs)
        m = self.models[model]

        if model == "bm25":
            Bw = _bm25_transform(m["cvw"], texts, m["idf_w"], m["avg_w"], self.k1, self.b)
            Bc = _bm25_transform(m["cvc"], texts, m["idf_c"], m["avg_c"], self.k1, self.b)
            X = hstack([Bw, Bc, D], format="csr")
            scores = m["clf"].decision_function(X)
        elif model == "nbsvm":
            Xb = binarize(hstack([m["w"].transform(texts), m["c"].transform(texts), D], format="csr"))
            X = Xb.multiply(m["R"]).tocsr()
            scores = m["clf"].decision_function(X)
        elif model == "linearsvm":
            X = hstack([m["w"].transform(texts), m["c"].transform(texts), D], format="csr")
            scores = m["clf"].decision_function(X)
        else:  # pragma: no cover
            raise KeyError(model)

        probs = _softmax(np.asarray(scores))
        pred_idx = probs.argmax(axis=1)
        preds = [self.labels[i] for i in pred_idx]
        score_df = pd.DataFrame(probs, columns=self.labels)
        return {
            "pred": preds,
            "confidence": probs.max(axis=1),
            "score_df": score_df,
            "model": model,
        }


if __name__ == "__main__":  # quick CLI smoke test
    p = TransactionsPredictor()
    samples = [
        "[debit] SQ *COFFEE SHOP #1234 portland OR 97201",
        "[credit] PYPL TRANSFER 44.10 ref 88321",
        "[debit] NETFLIX.COM RECURRING 15.49",
        "[debit] UBER TRIP HELP.UBER.COM",
    ]
    for mk in p.available_models():
        r = p.predict(samples, mk)
        print(f"\n[{mk}]")
        for t, pr, cf in zip(samples, r["pred"], r["confidence"]):
            print(f"  {cf:.3f}  {pr:16s} <- {t[:48]}")
