# dashboard
import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# page + layout
st.set_page_config(page_title="MRP Legal Models Dashboard", layout="wide")
st.markdown(
    """
<style>
/* cap width so charts don't stretch on ultra-wide screens */
.block-container { max-width: 1100px; padding-top: 0.8rem; padding-bottom: 1rem; }
.kpi { background:#0F172A10; border:1px solid #E5EAF2; border-radius:12px; padding:16px; text-align:center;}
.kpi .label { color:#6B7A90; font-size:0.95rem; }
.kpi .value { font-size:1.6rem; font-weight:700; }
</style>
""",
    unsafe_allow_html=True,
)
FIG_H = 340

# config
CSV_PATH = "evaluation_results.csv"
W_CLS = 0.5     # classification (LEDGAR, ContractNLI)
W_QA  = 0.5     # QA (CUAD)

NUM_FMT = ".3f"
def _fmt(x): return f"{x:.3f}" if pd.notna(x) else "—"

EXPECTED = ["Model","Accuracy",
            "Precision_micro","Recall_micro","F1_micro",
            "Precision_macro","Recall_macro","F1_macro",
            "EM","F1","F1_unified"]

C_NUM = ["Accuracy","Precision_micro","Recall_micro","F1_micro",
         "Precision_macro","Recall_macro","F1_macro","EM","F1","F1_unified"]

def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Model" in df.columns:
        df["Model"] = df["Model"].astype(str).str.strip()
        df = df[df["Model"].str.len() > 0]
        df = df[~df["Model"].str.lower().isin(["nan","none","null"])]
    for c in C_NUM:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_results(p=CSV_PATH):
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    for c in EXPECTED:
        if c not in df.columns:
            df[c] = np.nan
    return clean_results(df)

def find_col(df: pd.DataFrame, target: str):
    for c in df.columns:
        if c.strip().lower() == target.strip().lower():
            return c
    return None

def safe_idxmax(s: pd.Series):
    s = s.dropna()
    return None if s.empty else s.idxmax()

def compute_unified(df: pd.DataFrame, w_cls: float, w_qa: float) -> pd.Series:
    """
    Row-wise unified score:
      - If model name is 'cuad' -> use F1 (QA)
      - Else -> use F1_micro (classification)
    Then apply weights; renormalize if a side is missing.
    """
    f1_micro_col = find_col(df, "F1_micro")
    f1_qa_col    = find_col(df, "F1")

    out = {}
    for _, row in df.iterrows():
        model = str(row["Model"]).strip()
        is_qa = (model.lower() == "cuad")
        cls_val = row.get(f1_micro_col, np.nan) if f1_micro_col else np.nan
        qa_val  = row.get(f1_qa_col, np.nan)    if f1_qa_col    else np.nan
        has_c = pd.notna(cls_val)
        has_q = pd.notna(qa_val)
        wsum = (w_cls if has_c else 0) + (w_qa if has_q else 0)
        if wsum == 0:
            out[model] = np.nan
        else:
            out[model] = ((cls_val if has_c else 0) * (w_cls/wsum)
                        + (qa_val  if has_q else 0) * (w_qa/wsum))
    return pd.Series(out)

# charts
def fig_bar_unified(unified: pd.Series) -> go.Figure:
    d = unified.dropna().reset_index()
    if d.empty:
        return go.Figure()
    d.columns = ["Model","F1"]
    d = d.sort_values("F1", ascending=False)
    fig = px.bar(d, x="Model", y="F1", text=d["F1"].map(_fmt), title="Unified F1 (bar)")
    fig.update_traces(textposition="outside")
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[0,1.05], tickformat=".2f")
    fig.update_layout(plot_bgcolor="white", height=FIG_H, margin=dict(l=10,r=10,t=56,b=10))
    return fig

def fig_bar_f1_micro_macro(df: pd.DataFrame) -> go.Figure:
    sub = df[["Model","F1_micro","F1_macro"]].dropna()
    if sub.empty: return go.Figure()
    long = sub.melt(id_vars="Model", var_name="Metric", value_name="Score")
    long["Metric"] = long["Metric"].str.replace("_"," ").str.title()
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                 text=long["Score"].map(_fmt), title="F1 — Micro vs Macro (bars)")
    fig.update_traces(textposition="outside")
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[0,1.05], tickformat=".2f")
    fig.update_layout(plot_bgcolor="white", height=FIG_H,
                      margin=dict(l=10,r=10,t=56,b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def fig_bar_cls_micro(df: pd.DataFrame) -> go.Figure:
    sub = df[["Model","Precision_micro","Recall_micro","F1_micro"]].dropna()
    if sub.empty: return go.Figure()
    long = sub.melt(id_vars="Model", var_name="Metric", value_name="Score")
    long["Metric"] = long["Metric"].str.replace("_micro","",regex=False).str.title()
    fig = px.bar(long, x="Model", y="Score", color="Metric", barmode="group",
                 text=long["Score"].map(_fmt), title="Classification (micro) — P / R / F1 (bars)")
    fig.update_traces(textposition="outside")
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[0,1.05], tickformat=".2f")
    fig.update_layout(plot_bgcolor="white", height=FIG_H,
                      margin=dict(l=10,r=10,t=56,b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def fig_lollipop_unified(series_unified: pd.Series, df: pd.DataFrame) -> go.Figure:
    series_unified = series_unified.dropna()
    if series_unified.empty: return go.Figure()
    f1c, f1q = df.set_index("Model")["F1_micro"], df.set_index("Model")["F1"]
    order = list(series_unified.sort_values().index)
    xv = [series_unified[m] for m in order]
    fig = go.Figure()
    for m in order:
        fig.add_trace(go.Scatter(x=[0, series_unified[m]], y=[m, m], mode="lines",
                                 line=dict(color="rgba(120,140,160,0.35)", width=8),
                                 showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=xv, y=order, mode="markers+text",
        text=[_fmt(v) for v in xv], textposition="middle right",
        marker=dict(size=14, line=dict(color="white", width=1)),
        name="Unified F1",
        hovertext=[f"{m}<br>Unified: {_fmt(series_unified[m])}"
                   f"<br>F1\u2009micro: {_fmt(f1c.get(m,np.nan))}"
                   f"<br>QA\u2009F1: {_fmt(f1q.get(m,np.nan))}" for m in order],
        hoverinfo="text"
    ))
    fig.update_layout(title="Unified Performance Index (F1)",
        xaxis=dict(range=[0,1.02], tickformat=".2f", gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title=""),
        plot_bgcolor="white", height=FIG_H, margin=dict(l=10,r=10,t=56,b=10))
    return fig

def fig_bump_rank(df: pd.DataFrame) -> go.Figure:
    metrics = {
        "F1 micro": df.set_index("Model")["F1_micro"],
        "F1 macro": df.set_index("Model")["F1_macro"],
        "Accuracy": df.set_index("Model")["Accuracy"],
        "QA F1":    df.set_index("Model")["F1"],
    }
    order = ["F1 micro","F1 macro","Accuracy","QA F1"]
    rows=[]
    for mname in order:
        s=metrics[mname].dropna(); r=s.rank(ascending=False, method="dense")
        rows += [{"Model":m,"Metric":mname,"Rank":int(r[m]),"Score":v} for m,v in s.items()]
    dd = pd.DataFrame(rows)
    if dd.empty: return go.Figure()
    fig = go.Figure()
    for m, sub in dd.groupby("Model"):
        sub=sub.set_index("Metric").reindex(order).dropna()
        if sub.empty: continue
        fig.add_trace(go.Scatter(x=sub.index, y=sub["Rank"], mode="lines+markers", name=m,
            hovertext=[f"{m}<br>{k}: {_fmt(v)} (rank {int(rr)})" for k,v,rr in zip(sub.index, sub["Score"], sub["Rank"])],
            hoverinfo="text"))
    fig.update_layout(
        title="Bump Chart — Rank Across Metrics (1 = best)",
        yaxis=dict(autorange="reversed", dtick=1, gridcolor="rgba(0,0,0,0.06)"),
        xaxis=dict(type="category", gridcolor="rgba(0,0,0,0.06)"),
        plot_bgcolor="white", legend=dict(orientation="h", y=-0.22),
        height=FIG_H, margin=dict(l=40,r=10,t=56,b=56)
    )
    return fig

def fig_slope(df: pd.DataFrame) -> go.Figure:
    sub=df[["Model","F1_micro","F1_macro"]].dropna()
    if sub.empty: return go.Figure()
    fig=go.Figure()
    for _,r in sub.iterrows():
        fig.add_trace(go.Scatter(x=["F1 macro","F1 micro"], y=[r["F1_macro"],r["F1_micro"]],
            mode="lines+markers+text", text=[_fmt(r["F1_macro"]),_fmt(r["F1_micro"])],
            textposition="top center", name=r["Model"], marker=dict(size=8), line=dict(width=3)))
    fig.update_layout(title="F1 Macro → F1 Micro (Slopegraph)",
        yaxis=dict(range=[0,1.02], tickformat=".2f", gridcolor="rgba(0,0,0,0.06)"),
        xaxis=dict(gridcolor="rgba(0,0,0,0.06)"),
        plot_bgcolor="white", height=FIG_H, margin=dict(l=40,r=10,t=56,b=10))
    return fig

def fig_heatmap(df: pd.DataFrame, macro: bool) -> go.Figure:
    cols=["Precision_macro","Recall_macro","F1_macro"] if macro else ["Precision_micro","Recall_micro","F1_micro"]
    title="Precision / Recall / F1 Heatmap (macro)" if macro else "Precision / Recall / F1 Heatmap (micro)"
    hm=df[["Model"]+cols].dropna()
    if hm.empty: return go.Figure()
    z=hm[cols].to_numpy(); x=[c.split("_")[0].title() for c in cols]; y=hm["Model"].tolist()
    fig=px.imshow(z, x=x, y=y, text_auto=".3f", color_continuous_scale="Viridis", origin="lower", aspect="auto", zmin=0, zmax=1)
    fig.update_traces(hovertemplate="%{y} • %{x}: %{z:.3f}<extra></extra>")
    fig.update_layout(title=title, xaxis_title="", yaxis_title="", coloraxis_colorbar=dict(title=""),
                      plot_bgcolor="white", height=FIG_H, margin=dict(l=10,r=10,t=56,b=10))
    return fig

def fig_cuad(df: pd.DataFrame) -> go.Figure:
    sub=df[["Model","EM","F1"]].dropna()
    if sub.empty: return go.Figure()
    long=sub.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig=px.bar(long, x="Model", y="Score", color="Metric", barmode="group", title="CUAD — EM vs F1")
    fig.update_traces(text=long["Score"].map(_fmt), textposition="outside")
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[0,1.05], tickformat=".2f")
    fig.update_layout(plot_bgcolor="white", height=FIG_H, margin=dict(l=10,r=10,t=56,b=10))
    return fig

def fig_acc_vs_f1micro(df: pd.DataFrame) -> go.Figure:
    sub=df[["Model","Accuracy","F1_micro"]].dropna()
    if sub.empty: return go.Figure()
    long=sub.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig=px.bar(long, x="Model", y="Score", color="Metric", barmode="group", title="Accuracy vs F1 (micro)")
    fig.update_traces(text=long["Score"].map(_fmt), textposition="outside")
    fig.update_xaxes(type="category")
    fig.update_yaxes(range=[0,1.05], tickformat=".2f")
    fig.update_layout(plot_bgcolor="white", height=FIG_H, margin=dict(l=10,r=10,t=56,b=10))
    return fig

def fig_gap(df: pd.DataFrame) -> go.Figure:
    g=df.dropna(subset=["F1_micro","F1_macro"]).copy()
    if g.empty: return go.Figure()
    g["gap"]=g["F1_micro"]-g["F1_macro"]; g=g.sort_values("gap", ascending=False)
    fig=px.bar(g, x="Model", y="gap", title="Micro–Macro F1 Gap (F1μ − F1macro)")
    fig.update_traces(text=g["gap"].map(_fmt), textposition="outside")
    fig.update_xaxes(type="category")
    fig.update_layout(plot_bgcolor="white", height=FIG_H, margin=dict(l=10,r=10,t=56,b=10))
    return fig

def fig_dumbbell(df: pd.DataFrame, which: str) -> go.Figure:
    a = find_col(df, f"{which}_macro")
    u = find_col(df, f"{which}_micro")
    if not a or not u: return go.Figure()
    sub = df[["Model", a, u]].dropna()
    if sub.empty: return go.Figure()
    order = sub.sort_values(u, ascending=False)["Model"].tolist()
    y = order
    x0 = [sub.set_index("Model").loc[m, a] for m in y]
    x1 = [sub.set_index("Model").loc[m, u] for m in y]
    fig = go.Figure()
    for yi, (aa, bb) in enumerate(zip(x0, x1)):
        fig.add_trace(go.Scatter(x=[aa, bb], y=[y[yi], y[yi]], mode="lines",
                                 line=dict(width=4), showlegend=False, hoverinfo="skip"))
    fig.add_scatter(x=x0, y=y, mode="markers", name="macro", marker=dict(size=10))
    fig.add_scatter(x=x1, y=y, mode="markers", name="micro", marker=dict(size=10))
    fig.update_layout(title=f"Dumbbell — {which.capitalize()} (macro → micro)",
                      xaxis=dict(range=[0,1.02], tickformat=".2f"),
                      yaxis=dict(categoryorder="array", categoryarray=y),
                      legend=dict(orientation="h", y=-0.22),
                      plot_bgcolor="white", height=FIG_H, margin=dict(l=110,r=10,t=56,b=40))
    return fig

def fig_pr_scatter(df: pd.DataFrame, agg: str) -> go.Figure:
    pcol = find_col(df, f"Precision_{agg}")
    rcol = find_col(df, f"Recall_{agg}")
    if not pcol or not rcol: return go.Figure()
    sub = df[["Model", pcol, rcol]].dropna()
    if sub.empty: return go.Figure()
    fig = px.scatter(sub, x=pcol, y=rcol, text="Model", title=f"Precision vs Recall Scatter ({agg})")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig.update_traces(textposition="top center",
                      hovertemplate="Model=%{text}<br>P=%{x:.3f}<br>R=%{y:.3f}<extra></extra>")
    fig.update_xaxes(range=[0,1.02], tickformat=".2f")
    fig.update_yaxes(range=[0,1.02], tickformat=".2f")
    fig.update_layout(plot_bgcolor="white", height=FIG_H, margin=dict(l=10,r=10,t=56,b=10))
    return fig

# app
st.title("MRP Results Dashboard")
st.caption("LEDGAR (classification) • ContractNLI (classification) • CUAD (extractive QA)")

if not os.path.exists(CSV_PATH):
    st.error(f"Missing `{CSV_PATH}` next to this script.")
    st.stop()

df = load_results(CSV_PATH)
unified = compute_unified(df, W_CLS, W_QA).dropna()

tab_overview, tab_deep = st.tabs(["Overview", "Deep Dive"])

with tab_overview:
    # KPIs
    c1,c2,c3 = st.columns(3)
    best = safe_idxmax(unified)
    c1.markdown(
        '<div class="kpi"><div class="label">Best overall F1 (unified)</div>'
        f'<div class="value">{(best if best else "—")}: {(_fmt(unified.max()) if not unified.dropna().empty else "—")}</div></div>',
        unsafe_allow_html=True
    )
    c2.markdown(
        f'<div class="kpi"><div class="label">Avg. Micro-F1 (classification)</div>'
        f'<div class="value">{_fmt(df["F1_micro"].mean())}</div></div>',
        unsafe_allow_html=True
    )
    if df["EM"].notna().any():
        best_em_row = df.loc[df["EM"].idxmax()]
        c3.markdown(
            f'<div class="kpi"><div class="label">Best CUAD EM</div>'
            f'<div class="value">{best_em_row["Model"]}: {_fmt(best_em_row["EM"])}</div></div>',
            unsafe_allow_html=True
        )
    else:
        c3.markdown('<div class="kpi"><div class="label">Best CUAD EM</div><div class="value">—</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Quick bars
    cqb1, cqb2 = st.columns((1,1))
    with cqb1: st.plotly_chart(fig_bar_unified(unified), use_container_width=True)
    with cqb2: st.plotly_chart(fig_bar_f1_micro_macro(df), use_container_width=True)

    # Executive visuals
    c1, c2 = st.columns((1,1))
    with c1: st.plotly_chart(fig_lollipop_unified(unified, df), use_container_width=True)
    with c2: st.plotly_chart(fig_bump_rank(df), use_container_width=True)

    c3, c4 = st.columns((1,1))
    with c3: st.plotly_chart(fig_slope(df), use_container_width=True)
    with c4: st.plotly_chart(fig_heatmap(df, macro=False), use_container_width=True)

    c5, c6 = st.columns((1,1))
    with c5: st.plotly_chart(fig_cuad(df), use_container_width=True)
    with c6: st.plotly_chart(fig_acc_vs_f1micro(df), use_container_width=True)

    st.plotly_chart(fig_gap(df), use_container_width=True)

with tab_deep:
    st.subheader("Where gaps happen & balance checks")
    r1c1, r1c2 = st.columns((1,1))
    with r1c1: st.plotly_chart(fig_dumbbell(df, which="F1"), use_container_width=True)
    with r1c2: st.plotly_chart(fig_heatmap(df, macro=True), use_container_width=True)

    r2c1, r2c2 = st.columns((1,1))
    with r2c1: st.plotly_chart(fig_pr_scatter(df, agg="micro"), use_container_width=True)
    with r2c2: st.plotly_chart(fig_pr_scatter(df, agg="macro"), use_container_width=True)

    st.plotly_chart(fig_bar_cls_micro(df), use_container_width=True)
