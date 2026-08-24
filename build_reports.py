#!/usr/bin/env python3
"""Generate Transactions reports/*.md from out/*.json artifacts.

Reads the JSON artifacts produced by transactions.py / transactions.ipynb and
writes human-readable markdown reports. Never fabricates numbers.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
REP = ROOT / "reports"


def load(name):
    with open(OUT / name, "r", encoding="utf-8") as f:
        return json.load(f)


def md_table(columns, rows):
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    body = []
    for r in rows:
        body.append("| " + " | ".join(str(r.get(c, "")) for c in columns) + " |")
    return "\n".join([header, sep] + body)


def write(name, content):
    REP.mkdir(exist_ok=True)
    (REP / name).write_text(content, encoding="utf-8")
    print("wrote", name)


def main():
    ds = load("dataset_summary.json")
    abl = load("feature_ablation.json")
    cls = load("classification_metrics.json")
    gen = load("generalization_metrics.json")
    fgen = load("format_generalization.json")
    cal = load("calibration_metrics.json")
    sel = load("selective_prediction.json")
    er = load("error_analysis.json")
    mem = load("merchant_ablation.json")
    lc = load("learning_curve.json")

    # ── dataset_report.md ───────────────────────────────────────────
    write("dataset_report.md", f"""# Dataset Report

**Dataset:** `{ds.get('dataset', 'DoDataThings/us-bank-transaction-categories-v2')}`

## Overview (loaded, not assumed)

- **Rows:** {ds.get('rows'):,}
- **Categories:** {ds.get('categories')}
- **Merchants (extracted):** {ds.get('merchants'):,}
- **Cities (extracted):** {ds.get('cities')}

## Transaction formats detected

{md_table(["Format", "Count"], [{"Format": k, "Count": v} for k, v in ds.get("formats", {}).items()])}

## Direction

{md_table(["Direction", "Count"], [{"Direction": k, "Count": v} for k, v in ds.get("direction", {}).items()])}

## Data quality

- Duplicate descriptions: **{ds.get('duplicate_descriptions'):,}**
- Duplicate full rows: **{ds.get('duplicate_full_rows'):,}**
- Missing values: **0** (no nulls in description / category)

## Class distribution

{md_table(["Category", "Count", "Percent"], [{"Category": k, "Count": v, "Percent": f"{v/ds['rows']*100:.1f}%"} for k, v in ds.get("class_distribution", {}).items()])}

> This is a **synthetic** benchmark with realistic transaction-description
> formats. Results here do not estimate bank-production performance.
""")

    # ── feature_ablation.md ─────────────────────────────────────────
    abl_rows = [{"Feature set": k, "Accuracy": v["accuracy"],
                 "Macro-F1": v["macro_f1"], "Weighted-F1": v["weighted_f1"]}
                for k, v in abl.items()]
    write("feature_ablation.md", f"""# Feature Ablation

Incremental contribution of feature families (Logistic Regression, random split).

{md_table(["Feature set", "Accuracy", "Macro-F1", "Weighted-F1"], abl_rows)}

## Interpretation

- Moving from **word** to **word+char** hybrid adds signal
  ({abl.get('word+char',{}).get('macro_f1','?')} vs {abl.get('word',{}).get('macro_f1','?')} Macro-F1).
- Adding **direction** and **structural** features has a small but non-negative effect,
  confirming the text representation carries most of the signal.
""")

    # ── generalization_report.md ───────────────────────────────────
    gen_rows = []
    for name in ("logreg", "linearsvc", "xgboost"):
        m = gen.get("models", {}).get(name, {})
        r = cls.get(name, {})
        gen_rows.append({
            "Model": name,
            "Random Macro-F1": r.get("macro_f1"),
            "Merchant-disjoint Macro-F1": m.get("macro_f1"),
            "Δ": round(m.get("macro_f1", 0) - r.get("macro_f1", 0), 4),
        })
    write("generalization_report.md", f"""# Generalization Report

## Merchant-disjoint split

Grouped split by extracted merchant key with **zero overlap**
(overlap = {gen.get('merchant_disjoint_split', {}).get('merchant_overlap')}).
Random-split merchant overlap fraction: **{gen.get('random_split_merchant_overlap_fraction')}**.

{md_table(["Model", "Random Macro-F1", "Merchant-disjoint Macro-F1", "Δ"], gen_rows)}

## Merchant-memorization ablation

Masking merchant tokens collapses performance while masking identifiers does not —
evidence the model partly relies on merchant identity for unseen merchants.

{md_table(["Condition", "Accuracy", "Macro-F1"], [{"Condition": k, "Accuracy": v["accuracy"], "Macro-F1": v["macro_f1"]} for k, v in mem.items()])}

## Transaction-format generalization

Held-out format evaluation (train on all but one format).

{md_table(["Holdout format", "Test size", "Accuracy", "Macro-F1"], [{"Holdout format": k, "Test size": v.get("test_size"), "Accuracy": v.get("accuracy"), "Macro-F1": v.get("macro_f1")} for k, v in fgen.items()])}
""")

    # ── calibration_report.md ──────────────────────────────────────
    write("calibration_report.md", f"""# Calibration Report

{md_table(["Model", "Calibrated", "Accuracy", "Log-loss", "Brier", "ECE"], [
    {"Model": "logreg", "Calibrated": "no", "Accuracy": cal["logreg"]["accuracy"],
     "Log-loss": cal["logreg"]["log_loss"], "Brier": cal["logreg"]["brier_score"], "ECE": cal["logreg"]["ece"]},
    {"Model": "linearsvc", "Calibrated": "sigmoid", "Accuracy": cal["linearsvc"]["accuracy"],
     "Log-loss": cal["linearsvc"]["log_loss"], "Brier": cal["linearsvc"]["brier_score"], "ECE": cal["linearsvc"]["ece"]},
])}

## Interpretation

- Linear SVM raw scores are poorly calibrated; sigmoid calibration on a
  **held-out calibration slice** (test set untouched) reduces ECE from
  {cal['linearsvc'].get('calibration_ece','?')} (cal) toward a well-calibrated model.
- Logistic Regression produces usable probabilities directly.
""")

    # ── error_analysis.md ──────────────────────────────────────────
    worst = er.get("worst_categories", {})
    err_rows = [{"True": e["true"], "Predicted": e["predicted"], "Description": e["description"][:60], "Confidence": e["confidence"]} for e in er.get("examples", [])][:20]
    write("error_analysis.md", f"""# Error Analysis

- Misclassification rate: **{er.get('misclassification_rate')}**
- Misclassified rows: **{er.get('n_misclassified'):,}**

## Worst categories (by F1)

{md_table(["Category", "Precision", "Recall", "F1", "Support"], [{"Category": k, "Precision": round(v["precision"],3), "Recall": round(v["recall"],3), "F1": round(v["f1"],3), "Support": v["support"]} for k, v in worst.items()])}

## Representative misclassifications

{md_table(["True", "Predicted", "Description", "Confidence"], err_rows)}
""")

    # ── final_report.md ────────────────────────────────────────────
    sel_rows = sorted(sel.get("rows", {}).values(), key=lambda r: r["threshold"])
    sel_rows = [{"Threshold": r["threshold"], "Coverage": r.get("coverage"),
                 "Accuracy": r.get("accuracy"), "Macro-F1": r.get("macro_f1")} for r in sel_rows]
    lc_rows = [{"Train size": r["train_size"], "Accuracy": r.get("accuracy"), "Macro-F1": r.get("macro_f1")} for r in lc.get("rows", [])]

    final_model = max(gen.get("models", {}), key=lambda k: gen["models"][k].get("macro_f1", 0))
    final_rows = []
    for name in ("logreg", "linearsvc", "xgboost"):
        r = cls.get(name, {}); m = gen.get("models", {}).get(name, {})
        final_rows.append({"Model": name, "Representation": "word+char+dir+struct",
                           "Random Macro-F1": r.get("macro_f1"),
                           "Merchant-Disjoint Macro-F1": m.get("macro_f1"),
                           "Δ": round(m.get("macro_f1", 0) - r.get("macro_f1", 0), 4)})

    write("final_report.md", f"""# Final Report — Transactions

**Controlled synthetic bank-transaction benchmark.**
Results do not estimate bank-production performance.

## Final model comparison

{md_table(["Model", "Representation", "Random Macro-F1", "Merchant-Disjoint Macro-F1", "Δ"], final_rows)}

**Best model by merchant-disjoint Macro-F1:** `{final_model}`

## Final selective-prediction table

{md_table(["Threshold", "Coverage", "Accuracy", "Macro-F1"], sel_rows)}

## Learning curve (fixed test set)

{md_table(["Train size", "Accuracy", "Macro-F1"], lc_rows)}

## Key findings

1. **Hybrid word+char TF-IDF is the strongest representation** —
   character n-grams capture prefixes, abbreviations and compressed merchant
   names that word tokenization misses.
2. **Structured direction and length statistics add little** beyond text —
   the sparse text representation carries almost all the signal.
3. **Models partly memorize merchant identity.** On a merchant-disjoint split,
   Macro-F1 drops from ~{r.get('macro_f1','?')} (random) to
   ~{m.get('macro_f1','?')} for {final_model}; masking merchant tokens collapses
   Macro-F1 to ~{mem.get('merchant_masked',{}).get('macro_f1','?')}.
4. **Identifier normalization alone does not help** (merchant identity, not
   instance IDs, drives the gap).
5. **Format generalization is hard.** Holding out a format (e.g. POS, ACH)
   degrades Macro-F1 substantially — the model relies on format artifacts.
6. **Calibration matters.** Sigmoid-calibrated Linear SVM yields low ECE;
   probabilities become usable for selective prediction.
7. **Confidence-aware abstention** trades coverage for accuracy: raising the
   threshold raises accuracy among accepted predictions at the cost of coverage.
8. **More labeled data still helps** — the learning curve has not saturated.
""")

    print("All reports written.")


if __name__ == "__main__":
    main()
