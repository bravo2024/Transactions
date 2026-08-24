"""
Transactions - Bank Transaction Categorizer (standalone Streamlit app)

Run:
    streamlit run app.py

Loads the serving bundle at ./artifacts/transactions_linear_bundle.joblib (regenerate with
`python train_export.py`). Independent of the research notebook.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictor import TransactionsPredictor, MODEL_INFO  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "artifacts", "transactions_linear_bundle.joblib")
LEADERBOARD = os.path.join(HERE, "out", "leaderboard.json")

st.set_page_config(page_title="Transactions - Transaction Categorizer", layout="wide")

# ---------- visual system ----------
st.markdown(
    """
    <style>
      #MainMenu, footer, header {visibility: hidden;}
      section[data-testid="stSidebar"] {border-right: 1px solid #262730;}
      .block-container {padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1200px;}
      h1 {letter-spacing: -0.02em; margin-bottom: 0;}
      div[data-testid="stMetric"] {
          background: rgba(128,128,128,0.07); border: 1px solid rgba(128,128,128,0.18);
          border-radius: 10px; padding: 14px 16px 10px 16px;
      }
      div[data-testid="stMetricLabel"] p {font-size: 0.78rem; letter-spacing: 0.06em;
          text-transform: uppercase; opacity: 0.72;}
      .stButton > button {width: 100%; font-weight: 600;}
      div[data-testid="stExpander"] {border: 1px solid rgba(128,128,128,0.18);
          border-radius: 10px;}
      \.brand-eyebrow {font-size: 0.74rem; letter-spacing: 0.14em; text-transform:
          uppercase; opacity: 0.65; margin-bottom: -0.4rem;}
      \.brand-sub {opacity: 0.75; margin-top: 0.15rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading serving bundle")
def get_predictor():
    return TransactionsPredictor(BUNDLE)


@st.cache_data
def load_leaderboard():
    if os.path.exists(LEADERBOARD):
        try:
            return pd.read_json(LEADERBOARD)
        except Exception:
            return None
    return None


@st.cache_data
def load_json(name):
    path = os.path.join(HERE, "out", name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


try:
    predictor = get_predictor()
except FileNotFoundError as e:
    st.error("Serving bundle missing.")
    st.code(str(e))
    st.info("Regenerate it with: python train_export.py")
    st.stop()

bundle_meta = predictor.bundle.get("meta", {})
metrics = predictor.bundle.get("metrics", {})

def route_flag(confidence_values, threshold):
    import numpy as np
    return np.where(confidence_values >= threshold, "auto-accept", "review")


# ---------------- header ----------------
st.markdown('<p class="brand-eyebrow">Transactions &middot; Merchant-disjoint benchmark</p>',
            unsafe_allow_html=True)
st.title("Bank Transaction Categorizer")
st.markdown(
    '<p class="brand-sub">Sparse-linear models for 17-way transaction categorization. '
    'Top-ranked method: BM25 term weighting + linear SVM (Robertson &amp; Zaragoza 2009).</p>',
    unsafe_allow_html=True,
)
st.divider()

c1, c2, c3, c4 = st.columns(4)
best_name, best_stats = max(metrics.items(), key=lambda kv: kv[1]["macro_f1"])
c1.metric("Best macro-F1 (test)", f"{best_stats['macro_f1']:.4f}", best_name)
c2.metric("Models served", len(predictor.available_models()))
c3.metric("Unseen-merchant test set", f"{bundle_meta.get('n_test', 0):,}")
c4.metric("Category classes", len(predictor.labels))

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Model")
    keys = predictor.available_models()
    ordered = ["bm25", "nbsvm", "linearsvm"]
    ordered = [k for k in ordered if k in keys] + [k for k in keys if k not in ordered]
    chosen = st.radio("Estimator", ordered,
                      format_func=lambda k: MODEL_INFO.get(k, {}).get("name", k))
    card = predictor.model_card(chosen)
    if card.get("macro_f1") is not None:
        st.metric("Macro-F1 (test)", f"{card['macro_f1']:.4f}")
    st.caption(card.get("paper", ""))

    st.divider()
    tau = st.slider("Auto-accept confidence threshold", 0.0, 1.0, 0.70, 0.05,
                    help="Rows below this softmax-normalized score are flagged for review.")

    st.divider()
    with st.expander("Serving bundle"):
        st.json({"version": predictor.bundle.get("version"),
                 "created": predictor.bundle.get("created"),
                 "dataset": bundle_meta.get("dataset"),
                 "split": bundle_meta.get("split")}, expanded=False)

# ---------------- tabs ----------------
tab_pred, tab_batch, tab_board, tab_eval, tab_analysis, tab_method = st.tabs(
    ["Predict", "Batch scoring", "Leaderboard", "Evaluation", "Analysis",
     "Methodology"])

with tab_pred:
    txt = st.text_area(
        "Descriptions, one per line",
        value="[debit] SQ *COFFEE SHOP #1234 portland OR 97201\n"
              "[credit] PYPL TRANSFER 44.10 ref 88321\n"
              "[debit] NETFLIX.COM RECURRING 15.49",
        height=150,
        label_visibility="collapsed",
    )
    go = st.button("Score descriptions", type="primary")
    if go:
        lines = [l.strip() for l in txt.splitlines() if l.strip()]
        if not lines:
            st.warning("Enter at least one description.")
        else:
            r = predictor.predict(lines, chosen)
            shown = min(len(lines), 6)
            cols = st.columns(shown)
            for j, col in enumerate(cols):
                col.metric(f"Line {j+1}", r["pred"][j], f"{r['confidence'][j]:.1%}")
            st.divider()
            left, right = st.columns([1, 1])
            with left:
                st.subheader("Class scores")
                first = r["score_df"].iloc[0].sort_values(ascending=True).tail(8)
                fig = px.bar(x=first.values, y=first.index, orientation="h",
                             labels={"x": "normalized score", "y": ""})
                fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10),
                                  paper_bgcolor="rgba(0,0,0,0)",
                                  plot_bgcolor="rgba(0,0,0,0)", showlegend=False)
                fig.update_xaxes(range=[0, 1], gridcolor="rgba(128,128,128,0.2)")
                fig.update_yaxes(gridcolor="rgba(128,128,128,0.12)")
                st.plotly_chart(fig, use_container_width=True,
                                config={"displayModeBar": False})
            with right:
                st.subheader("All inputs")
                res = pd.DataFrame({
                    "description": [t[:64] for t in lines],
                    "predicted": r["pred"],
                    "confidence": r["confidence"],
                    "routing": ["auto-accept" if cf >= tau else "review"
                                for cf in r["confidence"]],
                })
                st.dataframe(
                    res,
                    column_config={
                        "description": st.column_config.TextColumn(width="medium"),
                        "confidence": st.column_config.ProgressColumn(
                            format="%.3f", min_value=0.0, max_value=1.0),
                    },
                    use_container_width=True, hide_index=True,
                )

with tab_batch:
    up = st.file_uploader("CSV containing a `description` column", type=["csv"])
    if up is not None:
        df_in = pd.read_csv(up)
        col = next((c for c in df_in.columns if c.lower() == "description"), None)
        if col is None:
            st.error("No `description` column found. Columns: "
                     + ", ".join(map(str, df_in.columns)))
        else:
            texts = df_in[col].astype(str).tolist()
            with st.spinner(f"Scoring {len(texts)} rows..."):
                r = predictor.predict(texts, chosen)
            out = df_in.copy()
            out["predicted_category"] = r["pred"]
            out["confidence"] = r["confidence"].round(4)
            out["routing"] = route_flag(out["confidence"].values, tau)
            flagged = int((out["routing"] == "review").sum())
            a, b, cc = st.columns(3)
            a.metric("Rows scored", len(out))
            b.metric("Auto-accepted", f"{len(out)-flagged:,}",
                     f"{100*(len(out)-flagged)/max(len(out),1):.1f}%")
            cc.metric("Routed to review", f"{flagged:,}")
            st.dataframe(out.head(100), use_container_width=True, hide_index=True)
            st.download_button("Download predictions (full)", 
                               data=out.to_csv(index=False).encode(),
                               file_name="transactions_predictions.csv", mime="text/csv")



with tab_board:
    lb = load_leaderboard()
    if lb is None:
        st.info("out/leaderboard.json not found; run the research notebook to generate it.")
    else:
        lb = lb.sort_values("Macro-F1", ascending=False).reset_index(drop=True)
        st.dataframe(
            lb[["Model", "Macro-F1", "Accuracy", "Train time (s)",
                "Inference latency (ms/sample)", "Size / complexity"]],
            column_config={
                "Model": st.column_config.TextColumn(width="large"),
                "Macro-F1": st.column_config.ProgressColumn(
                    format="%.4f", min_value=0.0, max_value=1.0),
                "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                "Train time (s)": st.column_config.NumberColumn(format="%.0f s"),
                "Inference latency (ms/sample)": st.column_config.NumberColumn(
                    format="%.4f ms"),
            },
            use_container_width=True, hide_index=True, height=420,
        )
        st.caption("Merchant-disjoint test split; lower-triangle of the study is the "
                   "deep-learning blend reported honestly.")


_HEAT_LAYOUT = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")

with tab_eval:
    st.subheader("Confusion matrix & per-class metrics")
    cm_data = load_json("classification_metrics.json")
    if not cm_data:
        st.info("out/classification_metrics.json not found; run the research notebook.")
    else:
        pretty = {"logreg": "Logistic Regression (calibrated)", "linearsvm": "Linear SVM",
                  "xgboost": "XGBoost"}
        sel = st.selectbox("Model", list(cm_data.keys()),
                           format_func=lambda k: pretty.get(k, k))
        entry = cm_data[sel]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Accuracy", f"{entry['accuracy']:.4f}")
        m2.metric("Macro-F1", f"{entry['macro_f1']:.4f}")
        m3.metric("Weighted-F1", f"{entry['weighted_f1']:.4f}")
        classes = [k for k in entry["classification_report"]
                   if k not in ("accuracy", "macro avg", "weighted avg")]
        report = pd.DataFrame(entry["classification_report"]).T.drop(
            index=["accuracy", "macro avg", "weighted avg"])

        left, right = st.columns([3, 2])
        with left:
            cm = np.asarray(entry["confusion_matrix"], dtype=float)
            fig = px.imshow(cm, x=classes, y=classes, text_auto=".0f",
                            color_continuous_scale="Blues", aspect="equal",
                            labels=dict(x="Predicted", y="True", color="count"))
            fig.update_layout(height=560, margin=dict(l=10, r=10, t=30, b=10),
                              coloraxis_showscale=False, **_HEAT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
        with right:
            st.markdown("**Per-class precision / recall / F1**")
            st.dataframe(
                report[["precision", "recall", "f1-score", "support"]],
                column_config={
                    "precision": st.column_config.NumberColumn(format="%.4f"),
                    "recall": st.column_config.NumberColumn(format="%.4f"),
                    "f1-score": st.column_config.NumberColumn(format="%.4f"),
                    "support": st.column_config.NumberColumn(format="%.0f"),
                },
                use_container_width=True, hide_index=True, height=520,
            )

    st.divider()
    st.subheader("Probability calibration")
    calib = load_json("calibration_metrics.json")
    final_calib = load_json("final_calibration_metrics.json")
    if calib:
        rows = []
        for fam, c in calib.items():
            rows.append({
                "model": fam,
                "log loss": c.get("log_loss"),
                "brier": c.get("brier_score"),
                "ECE": c.get("ece"),
                "raw ECE (uncalibrated)": c.get("raw_ece"),
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("Sigmoid (Platt) calibration via CalibratedClassifierCV; "
                   "ECE = expected calibration error (lower is better).")
        curve_sel = st.selectbox(
            "Calibration curve", list(calib.keys()),
            format_func=lambda k: pretty.get(k, k) if calib else k)
        cc = calib[curve_sel].get("calibration_curve")
        if cc:
            cdf = pd.DataFrame(cc)
            fig = px.line(cdf, x="bin_center", y=["confidence", "accuracy"],
                          markers=True)
            for tr, name in zip(fig.data, ["mean predicted confidence",
                                           "observed accuracy"]):
                tr.name = name
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"),
                          label=dict(text="perfect", textposition="bottom right"))
            fig.update_layout(height=320, xaxis_title="predicted probability bin",
                              yaxis_range=[0, 1], legend_title_text="",
                              margin=dict(l=10, r=10, t=30, b=10), **_HEAT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
    if final_calib:
        f1c, f2c, f3c = st.columns(3)
        f1c.metric("Deployed pipeline log loss (g_test)",
                   f"{final_calib['log_loss']:.4f}")
        f2c.metric("Brier", f"{final_calib['brier']:.4f}")
        f3c.metric("ECE", f"{final_calib['ece']:.4f}")

    st.divider()
    st.subheader("Error analysis")
    err = load_json("error_analysis.json")
    if not err:
        st.info("out/error_analysis.json not found.")
    else:
        e1, e2 = st.columns(2)
        e1.metric("Misclassified (test)", err["n_misclassified"])
        e2.metric("Misclassification rate", f"{err['misclassification_rate']:.2%}")
        worst = pd.DataFrame(err["worst_categories"]).T
        best = pd.DataFrame(err["best_categories"]).T
        wl, bl = st.columns(2)
        with wl:
            st.markdown("**Worst categories**")
            st.dataframe(worst.round(4), use_container_width=True, hide_index=True)
        with bl:
            st.markdown("**Best categories**")
            st.dataframe(best.round(4), use_container_width=True, hide_index=True)
        ex = pd.DataFrame(err["examples"])
        st.markdown(f"**Sample misclassified examples** ({len(ex)} shown)")
        st.dataframe(ex, use_container_width=True, hide_index=True)

with tab_analysis:
    st.subheader("Learning curve")
    lc = load_json("learning_curve.json")
    if lc:
        ldf = pd.DataFrame(lc["rows"])
        fig = px.line(ldf, x="train_size", y=["accuracy", "macro_f1"], markers=True,
                      labels={"value": "score", "variable": "metric",
                              "train_size": "training rows"})
        fig.update_layout(height=340, yaxis_range=[0.85, 1.0], legend_title_text="",
                          margin=dict(l=10, r=10, t=30, b=10), **_HEAT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

    left_a, right_a = st.columns(2)
    with left_a:
        st.subheader("Feature ablation")
        fa = load_json("feature_ablation.json")
        if fa:
            fadf = pd.DataFrame(fa).T.reset_index(names="features")
            fig = px.bar(fadf, x="features", y="macro_f1",
                         labels={"macro_f1": "Macro-F1", "features": ""})
            fig.update_layout(height=320, yaxis_range=[0.95, 1.0],
                              margin=dict(l=10, r=10, t=30, b=10), **_HEAT_LAYOUT)
            fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)")
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})
    with right_a:
        st.subheader("Novelty / OOD screening")
        nov = load_json("novelty_summary.json")
        if nov:
            fr = nov["flag_rate"]
            n1, n2, n3 = st.columns(3)
            n1.metric("IsoForest flag rate", f"{fr['isolation_forest_full']:.2%}")
            n2.metric("SGD-OCSVM flag rate", f"{fr['sgd_ocsvm_rff_full']:.2%}")
            agree = pd.DataFrame(nov["agreement"])
            st.dataframe(agree, use_container_width=True, hide_index=True)
            st.caption(str(nov.get("recommendation", "")))

    st.divider()
    st.subheader("Prescriptive routing sweep (confidence threshold)")
    rt = load_json("routing_table.json")
    if rt:
        rdf = pd.DataFrame(rt)
        rdf.columns = [c.replace("_", " ") for c in rdf.columns]
        st.dataframe(rdf, use_container_width=True, hide_index=True)
        fig = px.line(rdf, x="threshold", y=["coverage", "auto accept accuracy"],
                      markers=True)
        for i, name in enumerate(["coverage", "auto accept accuracy"]):
            fig.data[i].name = name
        fig.update_layout(height=320, yaxis_range=[0.6, 1.0], legend_title_text="",
                          margin=dict(l=10, r=10, t=30, b=10), **_HEAT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        st.caption("Operating point: highest threshold that keeps coverage >= 70% "
                   "(favor high auto-accept accuracy over coverage).")

    st.divider()
    st.subheader("Selective prediction (abstention)")
    sel_pred = load_json("final_selective_prediction.json")
    if not sel_pred:
        st.info("out/final_selective_prediction.json not found.")
    else:
        frozen = sel_pred["frozen_threshold_test"]
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Selected threshold", f"{sel_pred['selected_threshold']:.2f}")
        s2.metric("Coverage at threshold", f"{frozen['coverage']:.1%}")
        s3.metric("Accuracy on accepted", f"{frozen['accuracy']:.4f}")
        s4.metric("Abstain -> review", f"{frozen['abstention_rate']:.1%}")
        sdf = pd.DataFrame(sel_pred["test_full_table"]).T
        fig = px.line(sdf, x="threshold", y=["coverage", "accuracy"],
                      markers=True)
        fig.update_layout(height=320, xaxis_title="confidence threshold",
                          yaxis_range=[0.6, 1.0], legend_title_text="",
                          margin=dict(l=10, r=10, t=30, b=10), **_HEAT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})
        sdf.columns = [c.replace("_", " ") for c in sdf.columns]
        st.dataframe(sdf, use_container_width=True, hide_index=True)
        st.caption("Calibration source: "
                   f"{sel_pred.get('calibration_source', 'g_cal')}. Below the "
                   "threshold the pipeline abstains and routes to human review.")

    st.subheader("Dataset overview")
    ds = load_json("dataset_summary.json")
    if ds:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Rows", f"{ds['rows']:,}")
        d2.metric("Categories", ds["categories"])
        d3.metric("Unique merchants", f"{ds['merchants']:,}")
        d4.metric("Cities", ds["cities"])
        cd = pd.DataFrame(sorted(ds["class_distribution"].items(),
                                 key=lambda kv: kv[1]),
                          columns=["category", "rows"])
        fig = px.bar(cd, x="rows", y="category", orientation="h",
                     labels={"category": ""})
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                          **_HEAT_LAYOUT)
        fig.update_xaxes(gridcolor="rgba(128,128,128,0.2)")
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False})

with tab_method:
    a, b = st.columns([2, 1])
    with a:
        st.markdown(
            "**Task.** Categorize bank transaction strings into 17 classes under a "
            "merchant-disjoint split: canonical merchant keys extracted by regex are "
            "grouped so that no merchant seen in training appears in test. This measures "
            "generalization to unseen merchants rather than memorization.\n\n"
            "**Why sparse linear wins.** TF-IDF word + character n-gram features carry a "
            "near-saturated keyword-to-category signal. Deep models trained from scratch "
            "(Char-CNN 0.8254, BiLSTM 0.6912) undergeneralize on unseen merchants, and "
            "every representation-level hybrid (fusion, ensembling, autoencoder latents, "
            "frozen transformer features) tested below the linear baseline."
        )
        st.markdown("**Papers implemented**")
        st.markdown(
            "- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance "
            "Framework: BM25 and Beyond.* FnTIR 3(4).\n"
            "- Wang, S. & Manning, C. D. (2012). *Baselines and Bigrams.* ACL 2012.\n"
            "- Liu, F.T. et al. (2008). *Isolation Forest.* ICDM (novelty screening, notebook sec. 10)."
        )
    with b:
        st.markdown("**Caveats**")
        st.markdown(
            "- Controlled synthetic benchmark; does not estimate real-bank production "
            "performance.\n"
            "- Confidences are softmax-normalized linear scores, not calibrated "
            "probabilities.\n"
            "- NER token scores in the notebook are against weak regex-derived labels."
        )

st.divider()
st.caption("Transactions - primary estimator BM25 (Robertson & Zaragoza 2009); secondary "
           "estimator NBSVM (Wang & Manning 2012). Scores are uncalibrated normalized "
           "decision values.")
