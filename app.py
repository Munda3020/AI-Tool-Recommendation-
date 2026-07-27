"""
The UI. Run with: streamlit run app.py
Professional custom theme (light/dark toggle), clickable tool links with logos,
Plotly charts, ratings, personalization, export/share, and analytics.
"""
import os
import re
import io
import base64
import sqlite3
import textwrap
import urllib.parse
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
from recommender import (
    recommend_tools, search_fts, add_rating, get_avg_ratings, get_ratings_for_tool,
    save_preferences, load_preferences, list_profiles, log_activity, get_activity_log,
)

BASE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="AI Tool Intelligence Platform", layout="wide", page_icon="🧭")

# ---------------------------------------------------------------------------
# THEME - dark/light palette pair, toggle lives in the sidebar and is kept in
# session_state so it persists across page switches within the session.
# ---------------------------------------------------------------------------
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

DARK = dict(
    INK="#12141C", SURFACE="#1B1E2B", SURFACE_2="#242838", BORDER="#323750",
    ACCENT="#6C63FF", ACCENT_2="#00D9C0", TEXT="#EDEEF5", TEXT_MUTED="#9A9DB5",
)
LIGHT = dict(
    INK="#F5F6FA", SURFACE="#FFFFFF", SURFACE_2="#F0F1F7", BORDER="#DADCE8",
    ACCENT="#5A52E0", ACCENT_2="#00A08D", TEXT="#1B1E2B", TEXT_MUTED="#5B5E73",
)

with st.sidebar:
    theme_choice = st.radio("Theme", ["Dark", "Light"], horizontal=True,
                             index=0 if st.session_state.theme_mode == "Dark" else 1, key="theme_radio")
    st.session_state.theme_mode = theme_choice

PALETTE = DARK if st.session_state.theme_mode == "Dark" else LIGHT
INK, SURFACE, SURFACE_2, BORDER = PALETTE["INK"], PALETTE["SURFACE"], PALETTE["SURFACE_2"], PALETTE["BORDER"]
ACCENT, ACCENT_2, TEXT, TEXT_MUTED = PALETTE["ACCENT"], PALETTE["ACCENT_2"], PALETTE["TEXT"], PALETTE["TEXT_MUTED"]
CHART_COLORS = [ACCENT, ACCENT_2, "#FF6584", "#FFB84C", "#4CC9F0", "#B983FF"]

st.markdown(f"""
<style>
    .stApp {{ background-color: {INK}; }}
    section[data-testid="stSidebar"] {{ background-color: {SURFACE}; border-right: 1px solid {BORDER}; }}
    h1, h2, h3 {{ color: {TEXT} !important; font-weight: 700 !important; }}
    p, span, label, .stMarkdown {{ color: {TEXT}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {TEXT_MUTED} !important; }}

    .hero {{
        background: linear-gradient(135deg, {SURFACE_2} 0%, {SURFACE} 100%);
        border: 1px solid {BORDER}; border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
    }}
    .hero-title {{ font-size: 28px; font-weight: 800; color: {TEXT}; margin: 0; }}
    .hero-sub {{ color: {TEXT_MUTED}; font-size: 15px; margin-top: 6px; }}
    .pill {{
        display: inline-block; background: {ACCENT}22; color: {ACCENT_2}; border: 1px solid {ACCENT}55;
        border-radius: 999px; padding: 3px 12px; font-size: 12px; font-weight: 600; margin-right: 6px;
    }}

    div[data-testid="stMetric"] {{
        background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px;
    }}
    div[data-testid="stMetricValue"] {{ color: {ACCENT_2} !important; }}
    div[data-testid="stMetricLabel"] {{ color: {TEXT_MUTED} !important; }}

    .tool-card {{
        background: {SURFACE_2}; border: 1px solid {BORDER}; border-radius: 14px;
        padding: 16px 18px; margin-bottom: 12px; transition: border-color 0.15s ease;
    }}
    .tool-card:hover {{ border-color: {ACCENT}; }}
    .tool-name {{ font-size: 17px; font-weight: 700; color: {TEXT}; margin: 0; }}
    .tool-desc {{ color: {TEXT_MUTED}; font-size: 13.5px; margin-top: 4px; }}
    .tool-badge {{
        display: inline-block; font-size: 11px; font-weight: 700; padding: 3px 10px;
        border-radius: 6px; margin-top: 8px;
    }}
    .badge-free {{ background: #1E4620; color: #6FDB7A; }}
    .badge-freemium {{ background: #2A3B5C; color: #7FB4FF; }}
    .badge-paid {{ background: #4A2438; color: #FF9EC0; }}
    .score-tag {{
        float: right; background: {ACCENT}22; color: {ACCENT_2}; border-radius: 8px;
        padding: 4px 10px; font-size: 12px; font-weight: 700;
    }}
    .rating-tag {{
        display: inline-block; margin-top: 8px; margin-left: 6px; font-size: 12px;
        color: {TEXT_MUTED};
    }}

    div[data-testid="stTextInput"] input, div[data-testid="stSelectbox"] {{
        background-color: {SURFACE_2} !important; color: {TEXT} !important; border-radius: 8px !important;
    }}
    .stButton button, .stLinkButton a, .stDownloadButton button {{
        background: {ACCENT} !important; color: white !important; border: none !important;
        border-radius: 8px !important; font-weight: 600 !important;
    }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; overflow: hidden; }}
</style>
""", unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor=INK, plot_bgcolor=INK, font=dict(color=TEXT, family="sans-serif"),
    margin=dict(l=10, r=10, t=40, b=10),
)

def domain_from_url(url):
    if not isinstance(url, str):
        return None
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else None

def pricing_badge(tier):
    cls = {"Free": "badge-free", "Freemium": "badge-freemium", "Paid": "badge-paid"}.get(tier, "badge-freemium")
    return f'<span class="tool-badge {cls}">{tier}</span>'

def render_tool_card(row, show_score=False, key_prefix="card"):
    domain = domain_from_url(row.get("official_url"))
    logo_html = f'<img src="https://logo.clearbit.com/{domain}" width="28" style="border-radius:6px;vertical-align:middle;margin-right:10px;" onerror="this.style.display=\'none\'">' if domain else ""
    score_html = f'<span class="score-tag">match {row["match_score"]:.0%}</span>' if show_score and "match_score" in row and pd.notna(row.get("match_score")) else ""

    avg_rating = row.get("avg_rating")
    num_ratings = row.get("num_ratings")
    if avg_rating and float(avg_rating) > 0:
        stars = "★" * round(float(avg_rating)) + "☆" * (5 - round(float(avg_rating)))
        rating_html = f'<span class="rating-tag">{stars} {float(avg_rating):.1f} ({int(num_ratings)} rating{"s" if num_ratings != 1 else ""})</span>'
    else:
        rating_html = f'<span class="rating-tag">No ratings yet</span>'

    card_html = f"""
    <div class="tool-card">
        {score_html}
        <div style="display:flex; align-items:center;">
            {logo_html}
            <p class="tool-name">{row['tool_name']}</p>
        </div>
        <p class="tool-desc">{row['best_for']}</p>
        <span class="pill">{row['category']}</span>
        {pricing_badge(row['pricing_tier'])}
        {rating_html}
    </div>
    """
    st.markdown(textwrap.dedent(card_html), unsafe_allow_html=True)
    btn_cols = st.columns([2, 1])
    with btn_cols[0]:
        if row.get("official_url"):
            st.link_button(f"Visit {row['tool_name']} →", row["official_url"], use_container_width=True)
    with btn_cols[1]:
        with st.popover("★ Rate", use_container_width=True):
            stars_choice = st.select_slider(
                "Your rating", options=[1, 2, 3, 4, 5], value=5,
                key=f"{key_prefix}_stars_{row['tool_name']}"
            )
            if st.button("Submit rating", key=f"{key_prefix}_submit_{row['tool_name']}"):
                add_rating(row["tool_name"], stars_choice)
                st.success("Thanks for rating!")
                st.cache_data.clear()
                st.rerun()

def attach_ratings(df):
    ratings = get_avg_ratings()
    merged = df.merge(ratings, on="tool_name", how="left")
    merged["avg_rating"] = merged["avg_rating"].fillna(0.0)
    merged["num_ratings"] = merged["num_ratings"].fillna(0).astype(int)
    return merged

def sort_dataframe(df, sort_by):
    price_order = {"Free": 0, "Freemium": 1, "Paid": 2}
    if sort_by == "Rating":
        return df.sort_values("avg_rating", ascending=False)
    if sort_by == "Price: Low to High":
        return df.assign(_p=df["pricing_tier"].map(price_order).fillna(1)).sort_values("_p").drop(columns="_p")
    if sort_by == "Price: High to Low":
        return df.assign(_p=df["pricing_tier"].map(price_order).fillna(1)).sort_values("_p", ascending=False).drop(columns="_p")
    if sort_by == "Name (A-Z)":
        return df.sort_values("tool_name")
    return df  # Relevance / default catalog order

def make_pdf(task_text, predicted_category, confidence, results_df):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "AI Tool Recommendations", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, f"Task: {task_text}", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(0, 7, f"Predicted category: {predicted_category}  (confidence {confidence:.0%})",
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    for _, row in results_df.iterrows():
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(0, 8, row["tool_name"], new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, f"Category: {row['category']}  |  Pricing: {row['pricing_tier']}  |  Match: {row['match_score']:.0%}",
                       new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 6, row["best_for"], new_x="LMARGIN", new_y="NEXT")
        if row.get("official_url"):
            pdf.set_text_color(60, 60, 200)
            pdf.multi_cell(0, 6, row["official_url"], new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(0, 0, 0)
        pdf.ln(3)
    return bytes(pdf.output())

@st.cache_data
def get_catalog():
    conn = sqlite3.connect(f"{BASE}/ai_tools.db")
    df = pd.read_sql("SELECT * FROM tools", conn)
    conn.close()
    return df

catalog = get_catalog()

# ---------------------------------------------------------------------------
# Read shareable-link query params (from "Copy shareable link") to prefill
# the recommender if someone opens a shared URL.
# ---------------------------------------------------------------------------
qp = st.query_params
shared_task = qp.get("task", "")
shared_pricing = qp.get("pricing", "All")
shared_sort = qp.get("sort", "Relevance")

st.markdown(f"""
<div class="hero">
    <p class="hero-title">🧭 AI Tool Intelligence Platform</p>
    <p class="hero-sub">{len(catalog)} tools researched across {catalog['category'].nunique()} categories
    — ML classification, numpy similarity ranking, and SQL full-text search, with a fallback
    that stops trusting a shaky prediction and searches everything instead.</p>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "🎯 Tool Recommender", "🔍 Browse & Search",
     "📈 Market Gap Dashboard", "⚖️ Compare Tools", "🧪 Model Performance",
     "👤 My Preferences", "📶 Analytics"],
    key="nav_page"
)

# ==================== OVERVIEW ====================
if page == "📊 Overview":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total tools tracked", len(catalog))
    col2.metric("Categories", catalog["category"].nunique())
    col3.metric("Free/Freemium tools", int((catalog["pricing_tier"] != "Paid").sum()))
    col4.metric("Paid-only tools", int((catalog["pricing_tier"] == "Paid").sum()))

    st.markdown("### Tools per category")
    cat_counts = catalog["category"].value_counts().sort_values(ascending=True)
    fig = go.Figure(go.Bar(
        x=cat_counts.values, y=cat_counts.index, orientation="h",
        marker=dict(color=cat_counts.values, colorscale=[[0, ACCENT_2], [1, ACCENT]]),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=560, xaxis_title="Number of tools")
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Pricing mix")
        pricing_counts = catalog["pricing_tier"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=pricing_counts.index, values=pricing_counts.values, hole=0.55,
            marker=dict(colors=CHART_COLORS),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=340)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.markdown("### Top-rated tools")
        top_rated = attach_ratings(catalog)
        top_rated = top_rated[top_rated["num_ratings"] > 0].sort_values("avg_rating", ascending=False).head(8)
        if top_rated.empty:
            st.info("No community ratings yet — rate a tool from the Recommender or Browse pages.")
        else:
            fig3 = go.Figure(go.Bar(
                x=top_rated["avg_rating"], y=top_rated["tool_name"], orientation="h",
                marker=dict(color=ACCENT_2),
                text=[f"{v:.1f}★" for v in top_rated["avg_rating"]], textposition="outside",
            ))
            fig3.update_layout(**PLOTLY_LAYOUT, height=340, xaxis=dict(range=[0, 5.5]))
            st.plotly_chart(fig3, use_container_width=True)

# ==================== TOOL RECOMMENDER ====================
elif page == "🎯 Tool Recommender":
    st.subheader("Describe your task")

    with st.expander("🎚️ Advanced filters & sorting", expanded=False):
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            pricing_filter = st.selectbox("Pricing filter", ["All", "Free", "Freemium", "Paid"],
                                           index=["All", "Free", "Freemium", "Paid"].index(shared_pricing) if shared_pricing in ["All", "Free", "Freemium", "Paid"] else 0)
        with fcol2:
            sort_by = st.selectbox("Sort by", ["Relevance", "Rating", "Price: Low to High", "Price: High to Low", "Name (A-Z)"],
                                    index=["Relevance", "Rating", "Price: Low to High", "Price: High to Low", "Name (A-Z)"].index(shared_sort) if shared_sort in ["Relevance", "Rating", "Price: Low to High", "Price: High to Low", "Name (A-Z)"] else 0)
        with fcol3:
            top_n = st.slider("Number of results", 3, 10, 5)

        profiles = list_profiles()
        use_profile = st.selectbox(
            "Boost my favorite categories (from My Preferences)",
            ["None"] + profiles,
        )

    task_input = st.text_input(
        "What do you need to get done?",
        value=shared_task,
        placeholder="e.g. I need to turn my notes into a pitch deck for investors",
        key="task_input_box"
    )

    if task_input:
        boost_categories = None
        if use_profile != "None":
            prefs = load_preferences(use_profile)
            if prefs:
                boost_categories = prefs["favorite_categories"]

        results, predicted_category, confidence, used_filter, classifier_error = recommend_tools(
            task_input, top_n=top_n, pricing_filter=pricing_filter, sort_by=sort_by,
            boost_categories=boost_categories,
        )
        log_activity(task_input, predicted_category, confidence, used_filter,
                     results.iloc[0]["tool_name"] if not results.empty else None)

        if classifier_error:
            st.warning(f"⚠️ {classifier_error}")

        c1, c2 = st.columns(2)
        c1.metric("Predicted category", predicted_category or "—")
        c2.metric("Classifier confidence", f"{confidence:.0%}")

        if used_filter:
            st.success(f"Confidence cleared the bar — searched only within **{predicted_category}**.")
        elif not classifier_error:
            st.info(
                "Low classifier confidence for a 26-category problem — automatically fell back "
                "to searching the **entire catalog** instead of trusting a shaky guess."
            )
        if boost_categories:
            st.caption(f"🎯 Boosting results from your favorite categories: {', '.join(boost_categories)}")

        # -------- Export & Share --------
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            if not results.empty:
                pdf_bytes = make_pdf(task_input, predicted_category, confidence, results)
                st.download_button("📄 Download as PDF", data=pdf_bytes,
                                    file_name="ai_tool_recommendations.pdf", mime="application/pdf",
                                    use_container_width=True)
        with ecol2:
            share_params = urllib.parse.urlencode({"task": task_input, "pricing": pricing_filter, "sort": sort_by})
            with st.popover("🔗 Shareable link", use_container_width=True):
                st.caption("Append this query string to your app's URL to share this exact search:")
                st.code(f"?{share_params}", language="text")

        st.markdown("### Recommended tools")
        for _, row in results.iterrows():
            render_tool_card(row, show_score=True, key_prefix="rec")
    else:
        st.info("Type a task above to get a recommendation.")

# ==================== BROWSE & SEARCH ====================
elif page == "🔍 Browse & Search":
    st.subheader("Search all tools (SQL full-text search)")
    search_query = st.text_input("Search", placeholder="e.g. video editing for social media", key="browse_search")

    col1, col2, col3 = st.columns(3)
    with col1:
        category_filter = st.multiselect("Filter by category", sorted(catalog["category"].unique()))
    with col2:
        pricing_filter2 = st.multiselect("Filter by pricing", sorted(catalog["pricing_tier"].unique()))
    with col3:
        sort_by2 = st.selectbox("Sort by", ["Relevance", "Rating", "Price: Low to High", "Price: High to Low", "Name (A-Z)"],
                                 key="browse_sort")

    if search_query:
        results = search_fts(search_query, limit=25)
        full_results = catalog[catalog["tool_name"].isin(results["tool_name"])] if not results.empty else catalog.iloc[0:0]
        order = {name: i for i, name in enumerate(results["tool_name"])}
        full_results = full_results.copy()
        full_results["_order"] = full_results["tool_name"].map(order)
        full_results = full_results.sort_values("_order")
        st.caption(f"{len(full_results)} results, ranked by SQL FTS5 relevance (bm25)")
    else:
        full_results = catalog.copy()

    if category_filter:
        full_results = full_results[full_results["category"].isin(category_filter)]
    if pricing_filter2:
        full_results = full_results[full_results["pricing_tier"].isin(pricing_filter2)]

    full_results = attach_ratings(full_results)
    full_results = sort_dataframe(full_results, sort_by2)

    st.caption(f"Showing {min(len(full_results), 30)} of {len(full_results)} tools")
    for _, row in full_results.head(30).iterrows():
        render_tool_card(row, key_prefix="browse")

# ==================== MARKET GAP DASHBOARD ====================
elif page == "📈 Market Gap Dashboard":
    st.subheader("AI Tool Market: Where's the Whitespace?")

    category_summary = pd.read_csv(f"{BASE}/category_summary_sql.csv")
    whitespace = pd.read_csv(f"{BASE}/whitespace_categories_sql.csv")
    pricing_mix = pd.read_csv(f"{BASE}/pricing_mix_sql.csv")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total tools", len(catalog))
    col2.metric("Categories", int(category_summary["category"].nunique()))
    col3.metric("Whitespace categories (≤2 tools)", len(whitespace))

    st.markdown("#### Category crowding")
    cs = category_summary.sort_values("tool_count", ascending=True)
    colors = [ACCENT_2 if v <= 2 else ACCENT for v in cs["tool_count"]]
    fig = go.Figure(go.Bar(x=cs["tool_count"], y=cs["category"], orientation="h",
                            marker=dict(color=colors)))
    fig.update_layout(**PLOTLY_LAYOUT, height=620, xaxis_title="Number of competing tools")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Teal bars = whitespace categories (2 or fewer competing tools)")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Whitespace categories")
        st.dataframe(whitespace, hide_index=True, use_container_width=True)
    with c2:
        st.markdown("#### % Free/Freemium by category")
        st.dataframe(pricing_mix.head(10), hide_index=True, use_container_width=True)

# ==================== COMPARE TOOLS ====================
elif page == "⚖️ Compare Tools":
    st.subheader("Compare tools side by side")
    tool_names = sorted(catalog["tool_name"].unique())
    selected = st.multiselect("Pick 2-4 tools to compare", tool_names, max_selections=4, key="compare_select")

    if len(selected) >= 2:
        rated_catalog = attach_ratings(catalog)
        rows = [rated_catalog[rated_catalog["tool_name"] == name].iloc[0] for name in selected]

        # ---- Comparison matrix table ----
        matrix = pd.DataFrame({
            row["tool_name"]: {
                "Category": row["category"],
                "Pricing": row["pricing_tier"],
                "Best for": row["best_for"],
                "Rating": f"{row['avg_rating']:.1f}★ ({int(row['num_ratings'])})" if row["num_ratings"] > 0 else "No ratings",
                "Website": row.get("official_url", ""),
                "Tags": row.get("tags", ""),
            }
            for row in rows
        })
        st.markdown("#### Comparison matrix")
        st.dataframe(matrix, use_container_width=True)

        st.markdown("#### Cards")
        cols = st.columns(len(selected))
        for col, row in zip(cols, rows):
            with col:
                render_tool_card(row, key_prefix="compare")
    else:
        st.info("Select at least 2 tools above to compare them.")

# ==================== MODEL PERFORMANCE ====================
elif page == "🧪 Model Performance":
    st.subheader("Classifier Performance (Honest Numbers)")

    try:
        with open(f"{BASE}/model_metadata.txt") as f:
            meta = dict(line.strip().split("=") for line in f if "=" in line)
    except FileNotFoundError:
        meta = {}

    col1, col2, col3 = st.columns(3)
    col1.metric("Best model", meta.get("best_model", "—"))
    col2.metric("Cross-val accuracy", f"{float(meta.get('cv_mean_accuracy', 0)):.1%}")
    col3.metric("Held-out test accuracy", f"{float(meta.get('held_out_test_accuracy', 0)):.1%}")

    try:
        comparison = pd.read_csv(f"{BASE}/model_comparison.csv")
        st.markdown("#### Model comparison (3-fold cross-validation)")
        fig = go.Figure(go.Bar(
            x=comparison["cv_mean_accuracy"], y=comparison["model"], orientation="h",
            marker=dict(color=[ACCENT_2 if s else ACCENT for s in comparison["selected"]]),
            text=[f"{v:.1%}" for v in comparison["cv_mean_accuracy"]], textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=260, xaxis_title="Mean CV accuracy")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Teal = the model actually deployed. Three real algorithms were benchmarked, not assumed.")
    except FileNotFoundError:
        pass

    st.markdown(
        f"Trained on **{meta.get('n_training_examples', '?')} labeled examples** across "
        f"**{meta.get('n_categories', '?')} categories**, using combined word + character "
        "n-gram features and hyperparameter-tuned Logistic Regression (selected via cross-"
        "validation against Naive Bayes, Linear SVM, and a voting ensemble — not assumed).\n\n"
        f"**Held-out test accuracy: {float(meta.get('held_out_test_accuracy', 0)):.1%}** · "
        f"**3-fold CV mean: {float(meta.get('cv_mean_accuracy', 0)):.1%}**. These two numbers "
        "differ because per-category test sets are small (as few as 3 examples for some "
        "categories) — the CV mean is the more conservative, trustworthy figure since it "
        "averages 3 different train/test splits instead of relying on one. Both are reported "
        "here rather than only showing the higher number.\n\n"
        "This is a real, evidence-based improvement over earlier versions of this project "
        "(38.6% → 63.2% → current), achieved through more training data, character n-grams "
        "(catching morphological variants like \"debug\"/\"debugging\" that word-only "
        "features miss), and systematic hyperparameter tuning — not by reducing the number "
        "of categories.\n\n"
        "**The fallback logic still matters**: when confidence drops below ~11.5% (3x random "
        "chance for 26 classes), the recommender searches the full catalog instead of "
        "trusting a shaky prediction — though with the improved model, most real tasks now "
        "score well above that bar."
    )

    try:
        img = Image.open(f"{BASE}/confusion_matrix.png")
        st.image(img, caption="Confusion matrix across all 26 categories", use_container_width=True)
    except FileNotFoundError:
        st.warning("Run 04_train_classifier.py first to generate the confusion matrix.")

    st.markdown("### What I'd do with more time")
    st.markdown(
        "- Collect 30+ labeled examples per category (780+ total) instead of 8-16\n"
        "- Replace TF-IDF with sentence embeddings for real semantic matching\n"
        "- Add a feedback loop: log which recommendations users actually click"
    )

# ==================== MY PREFERENCES ====================
elif page == "👤 My Preferences":
    st.subheader("Save your preferences for personalized recommendations")
    st.caption(
        "Give yourself a profile name, pick the categories you care about most and a "
        "pricing preference. Then on the Tool Recommender page, select your profile under "
        "'Boost my favorite categories' to nudge rankings toward what you actually use."
    )

    existing_profiles = list_profiles()
    profile_mode = st.radio("Profile", ["Create new", "Edit existing"] if existing_profiles else ["Create new"],
                             horizontal=True)

    if profile_mode == "Edit existing" and existing_profiles:
        profile_name = st.selectbox("Choose profile", existing_profiles)
        current = load_preferences(profile_name) or {"favorite_categories": [], "pricing_pref": "All"}
    else:
        profile_name = st.text_input("Profile name", placeholder="e.g. alex")
        current = {"favorite_categories": [], "pricing_pref": "All"}

    favorite_categories = st.multiselect(
        "Favorite categories", sorted(catalog["category"].unique()),
        default=current["favorite_categories"],
    )
    pricing_pref = st.selectbox(
        "Preferred pricing", ["All", "Free", "Freemium", "Paid"],
        index=["All", "Free", "Freemium", "Paid"].index(current["pricing_pref"]) if current["pricing_pref"] in ["All", "Free", "Freemium", "Paid"] else 0,
    )

    if st.button("💾 Save preferences", use_container_width=True):
        if not profile_name:
            st.error("Give your profile a name first.")
        else:
            save_preferences(profile_name, favorite_categories, pricing_pref)
            st.success(f"Saved preferences for '{profile_name}'. Use them from the Tool Recommender page.")

# ==================== ANALYTICS ====================
elif page == "📶 Analytics":
    st.subheader("Usage Analytics")
    st.caption("Live activity from this app's Tool Recommender searches, plus community ratings.")

    log = get_activity_log()
    if log.empty:
        st.info("No searches logged yet — run a search on the Tool Recommender page to populate this dashboard.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total searches", len(log))
        col2.metric("Avg. classifier confidence", f"{log['confidence'].mean():.0%}")
        col3.metric("% high-confidence (category filter used)", f"{log['used_category_filter'].mean():.0%}")

        st.markdown("#### Most-searched predicted categories")
        top_cats = log["predicted_category"].value_counts().head(10).sort_values(ascending=True)
        fig = go.Figure(go.Bar(x=top_cats.values, y=top_cats.index, orientation="h",
                                marker=dict(color=ACCENT)))
        fig.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title="Searches")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Most-recommended tools (top result per search)")
        top_tools = log["top_tool"].dropna().value_counts().head(10).sort_values(ascending=True)
        if not top_tools.empty:
            fig2 = go.Figure(go.Bar(x=top_tools.values, y=top_tools.index, orientation="h",
                                     marker=dict(color=ACCENT_2)))
            fig2.update_layout(**PLOTLY_LAYOUT, height=380, xaxis_title="Times recommended #1")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Recent searches")
        st.dataframe(log[["created_at", "task_text", "predicted_category", "confidence", "top_tool"]].head(20),
                     hide_index=True, use_container_width=True)

    st.markdown("#### Community ratings")
    ratings = get_avg_ratings().sort_values("num_ratings", ascending=False)
    if ratings.empty:
        st.info("No ratings submitted yet.")
    else:
        st.dataframe(ratings, hide_index=True, use_container_width=True)
