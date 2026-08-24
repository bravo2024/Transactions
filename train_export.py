"""
Transactions standalone serving-bundle trainer.

Retrains only the fast linear estimators (top-ranked BM25 pipeline, the NBSVM
variant, and the TF-IDF LinearSVM baseline) on the merchant-disjoint split and persists a single portable
joblib bundle that app.py / predictor.py load at inference time.

Usage:
    python train_export.py            # writes artifacts/transactions_linear_bundle.joblib

Independent of the research notebook — no notebook state required.
"""
import time, re, json, os
import numpy as np, pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder, binarize
from datasets import load_dataset
import joblib

RS = 42
HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")
K1, B = 1.5, 0.75

# ---------- text preprocessing (mirrors notebook §3) ----------
_WRAPPER_RE = re.compile(r"^\[(debit|credit)\]\s*|\bsq\s*\*|\btst\*|\bpypl\*|\bppd id:?\s*\S*|\bsig purchase\b|\bach debit\b|\bach credit\b|\bpos debit\b|\bpos purchase\b|\bdebit card purchase\b|\brecurring payment\b|\bwire transfer\b", re.I)
_LONGDIGIT_RE = re.compile(r"\b\d{4,}\b")
_REFSTORE_RE = re.compile(r"\b(ref|id|store|order|txn|conf)\.?#?\s*\S+\b", re.I)
_HASHNUM_RE = re.compile(r"#\s*\d+")
_STATEZIP_RE = re.compile(r"\b[A-Z]{2}\s*\d{5}(-\d{4})?\b")
_PUNCT_RE = re.compile(r"[^a-z0-9\s]")

def direction_of(desc: str) -> str:
    d = str(desc).strip().lower()
    return 'debit' if d.startswith('[debit]') else ('credit' if d.startswith('[credit]') else 'unknown')

DIR_COLS = ['debit', 'credit', 'unknown']

def direction_block(frame_or_texts):
    if hasattr(frame_or_texts, 'columns'):
        d = pd.get_dummies(frame_or_texts['direction']).reindex(columns=DIR_COLS, fill_value=0)
    else:
        d = pd.get_dummies([direction_of(t) for t in frame_or_texts]).reindex(columns=DIR_COLS, fill_value=0)
    return csr_matrix(d.values.astype(np.float64))

def bm25_from_counts(C, idf, avg_len, k1=K1, b=B):
    C = C.tocoo()
    L = np.asarray(C.sum(axis=1)).ravel()
    denom = C.data + k1 * (1 - b + b * L[C.row] / avg_len)
    vals = idf[C.col] * (C.data * (k1 + 1)) / denom
    return sp.csr_matrix((vals, (C.row, C.col)), shape=C.shape)

def main():
    t0 = time.time()
    print('loading dataset...', flush=True)
    df = load_dataset('DoDataThings/us-bank-transaction-categories-v2')['train'].to_pandas()
    df = df.dropna(subset=['description', 'category']).reset_index(drop=True)
    LABELS = sorted(df['category'].unique())
    le = LabelEncoder().fit(LABELS)
    df['direction'] = df['description'].apply(direction_of)
    def merchant(desc):
        s = _WRAPPER_RE.sub(' ', str(desc).strip()); s = _STATEZIP_RE.sub(' ', s); s = _HASHNUM_RE.sub(' ', s)
        s = _REFSTORE_RE.sub(' ', s); s = _LONGDIGIT_RE.sub(' ', s); s = s.lower(); s = _PUNCT_RE.sub(' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        t = [x for x in s.split(' ') if x and not x.isdigit()]
        return (' '.join(t[:2]) if len(t) > 1 else t[0]) if t else 'unknown'
    df['merchant'] = df['description'].apply(merchant)
    ms = df['merchant'].unique(); rng = np.random.RandomState(RS); rng.shuffle(ms)
    ts = set(ms[:max(1, int(len(ms) * 0.2))])
    it = df['merchant'].isin(ts)
    train_df, test_df = df.loc[~it].reset_index(drop=True), df.loc[it].reset_index(drop=True)
    y_tr, y_te = train_df['category'].values, test_df['category'].values
    print(f'split train={len(train_df)} test={len(test_df)} ({time.time()-t0:.1f}s)', flush=True)

    # ---- view builders ----
    w = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), sublinear_tf=True, max_features=15000).fit(train_df['description'])
    c = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), sublinear_tf=True, max_features=20000).fit(train_df['description'])
    D_tr, D_te = direction_block(train_df), direction_block(test_df)
    X_tr = hstack([w.transform(train_df['description']), c.transform(train_df['description']), D_tr], format='csr')
    X_te = hstack([w.transform(test_df['description']), c.transform(test_df['description']), D_te], format='csr')
    def mf(y, p): return float(f1_score := __import__('sklearn.metrics', fromlist=['f1_score']).f1_score(y, p, average='macro', labels=LABELS, zero_division=0))

    metrics = {}

    # ---- model 1: baseline LinearSVM on TF-IDF ----
    t0 = time.time()
    linsvc = LinearSVC(max_iter=1000, random_state=RS).fit(X_tr, y_tr)
    f_base = mf(y_te, linsvc.predict(X_te))
    metrics['Linear SVM (TF-IDF)'] = {'macro_f1': f_base, 'train_s': round(time.time() - t0, 1)}
    print(f'[1/3] LinearSVM TF-IDF      F1={f_base:.4f} ({time.time()-t0:.1f}s)', flush=True)

    # ---- model 2: NBSVM (Wang & Manning 2012) ----
    t0 = time.time()
    Xb_tr, Xb_te = binarize(X_tr), binarize(X_te)
    nf = Xb_tr.shape[1]; r_pc = []
    for cl in LABELS:
        m = (y_tr == cl)
        cnt = np.asarray(Xb_tr[m].sum(axis=0)).ravel()
        p = (1.0 + cnt) / (1.0 * nf + m.sum())
        r_pc.append(np.log(p / (1.0 - p)))
    R = np.mean(np.vstack(r_pc), axis=0)
    Xnb_tr, Xnb_te = Xb_tr.multiply(R).tocsr(), Xb_te.multiply(R).tocsr()
    nbsvm_clf = LinearSVC(max_iter=1000, random_state=RS).fit(Xnb_tr, y_tr)
    f_nb = mf(y_te, nbsvm_clf.predict(Xnb_te))
    metrics['NBSVM (Wang & Manning 2012)'] = {'macro_f1': f_nb, 'train_s': round(time.time() - t0, 1)}
    print(f'[2/3] NBSVM                 F1={f_nb:.4f} ({time.time()-t0:.1f}s)', flush=True)

    # ---- model 3: BM25 + LinearSVC (top-ranked) ----
    t0 = time.time()
    cvw = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=30000).fit(train_df['description'])
    cvc = CountVectorizer(analyzer='char', ngram_range=(3, 5), max_features=40000).fit(train_df['description'])
    def fit_stats(cv, texts):
        C = cv.transform(texts).tocsc()
        L = np.asarray(C.sum(axis=1)).ravel(); avg = float(L.mean())
        dfreq = np.asarray((C > 0).sum(axis=0)).ravel(); N = C.shape[0]
        idf = np.log((N - dfreq + 0.5) / (dfreq + 0.5) + 1e-9)
        return idf.astype(np.float64), avg
    idf_w, avg_w = fit_stats(cvw, train_df['description'])
    idf_c, avg_c = fit_stats(cvc, train_df['description'])
    Bw_tr = bm25_from_counts(cvw.transform(train_df['description']), idf_w, avg_w)
    Bw_te = bm25_from_counts(cvw.transform(test_df['description']), idf_w, avg_w)
    Bc_tr = bm25_from_counts(cvc.transform(train_df['description']), idf_c, avg_c)
    Bc_te = bm25_from_counts(cvc.transform(test_df['description']), idf_c, avg_c)
    Xb25_tr = hstack([Bw_tr, Bc_tr, D_tr], format='csr')
    Xb25_te = hstack([Bw_te, Bc_te, D_te], format='csr')
    bm25_clf = LinearSVC(max_iter=1000, random_state=RS).fit(Xb25_tr, y_tr)
    f_bm = mf(y_te, bm25_clf.predict(Xb25_te))
    metrics['BM25 + LinearSVC'] = {'macro_f1': f_bm, 'train_s': round(time.time() - t0, 1)}
    print(f'[3/3] BM25 + LinearSVC      F1={f_bm:.4f} ({time.time()-t0:.1f}s)', flush=True)

    bundle = {
        'version': 1,
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'labels': LABELS,
        'k1': K1, 'b': B,
        'models': {
            'bm25':    {'clf': bm25_clf, 'cvw': cvw, 'cvc': cvc, 'idf_w': idf_w, 'idf_c': idf_c,
                        'avg_w': avg_w, 'avg_c': avg_c},
            'nbsvm':   {'clf': nbsvm_clf, 'w': w, 'c': c, 'R': R},
            'linearsvm': {'clf': linsvc, 'w': w, 'c': c},
        },
        'metrics': metrics,
        'meta': {'split': 'merchant-disjoint (seed 42, 80/20 by canonical merchant)',
                 'dataset': 'DoDataThings/us-bank-transaction-categories-v2',
                 'n_train': int(len(train_df)), 'n_test': int(len(test_df))},
    }
    os.makedirs(ART, exist_ok=True)
    out = os.path.join(ART, 'transactions_linear_bundle.joblib')
    joblib.dump(bundle, out, compress=3)
    print('\nSUMMARY (test Macro-F1):')
    for k, v in sorted(metrics.items(), key=lambda kv: -kv[1]['macro_f1']):
        print(f'  {k:34s} {v["macro_f1"]:.4f}')
    print(f'\nsaved -> {out}')

if __name__ == '__main__':
    main()
