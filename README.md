# Transactions - Bank Transaction Categorization

Sparse-linear models for 17-way bank transaction categorization on a
merchant-disjoint split, with a standalone Streamlit scoring application.

The study compares classical baselines, research-paper feature weightings, and a
from-scratch deep-learning blend under one shared evaluation protocol. Feature
weighting - not architecture - drives the result: BM25 term weighting and NBSVM
log-ratio features outperform every neural variant tested.

## Results (merchant-disjoint test set)

| Model | Macro-F1 | Accuracy | Train time | Latency |
|---|---|---|---|---|
| BM25 + LinearSVC | **0.9650** | 0.9599 | 82 s | 0.0019 ms/sample |
| NBSVM (Wang & Manning 2012) | 0.9548 | 0.9464 | 15 s | 0.0012 ms/sample |
| Linear SVM (TF-IDF) | 0.9401 | 0.9336 | 58 s | 0.0014 ms/sample |
| Logistic Regression | 0.9117 | 0.9016 | 17 s | 0.0014 ms/sample |
| LightGBM | 0.8451 | 0.8327 | 423 s | 0.0341 ms/sample |
| Char-CNN (1D, from scratch) | 0.8254 | 0.8102 | 133 s | 0.1705 ms/sample |
| XGBoost | 0.7968 | 0.7866 | 660 s | 0.0128 ms/sample |
| BiLSTM + Attention | 0.6912 | 0.6654 | 331 s | 0.8439 ms/sample |

NER sub-task (weak supervision): token-level macro-F1 0.957.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# artifacts/transactions_linear_bundle.joblib is included; to regenerate it:
python train_export.py

streamlit run app.py
```

The application is fully independent of the research notebook: it loads one
serialized bundle containing the vectorizers and linear estimators.

## Repository structure

```
transactionbert.ipynb   Full study: data, features, models, novelty routing
app.py                  Streamlit scoring interface
predictor.py            Inference core (no Streamlit dependency)
train_export.py         Retrains and exports the serving bundle
artifacts/              Serialized serving bundle (joblib)
out/                    Metrics artifacts (leaderboard, novelty, routing)
```

## Methodology

- Dataset: `DoDataThings/us-bank-transaction-categories-v2` (68,000 rows,
  17 balanced classes). Controlled synthetic benchmark; results do not
  estimate real-bank production performance.
- Evaluation: canonical merchant keys are extracted by regex and grouped so no
  merchant in training appears in test (80/20 split, seed 42).
- Papers implemented:
  - Robertson & Zaragoza (2009), *BM25* - top-ranked model.
  - Wang & Manning (2012), *NBSVM* - Naive-Bayes log-count ratios into an SVM.
  - Liu et al. (2008), *Isolation Forest* - linear-time novelty screening.
- A pretrained-transformer classifier was evaluated and excluded: the target
  environment ships CPU-only torch, making both fine-tuning and frozen feature
  extraction impractical, with no accuracy benefit over the sparse models.

Confidence values shown in the application are softmax-normalized decision
scores, not calibrated probabilities.
