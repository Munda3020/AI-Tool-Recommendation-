"""
Shared recommender logic - imported by the Streamlit app.

DESIGN DECISION (worth explaining in an interview):
With 26 categories and as few as 4 training examples for some of them, the
classifier is unreliable for rare categories (see confusion_matrix.png -
several categories get 0% recall). Blindly trusting a low-confidence
prediction would filter out the correct tool before it's ever ranked.

So: if the classifier's confidence is high, narrow the search to that
category (faster, more precise). If confidence is low, skip the category
filter entirely and rank across the FULL catalog instead. This means a
weak classifier degrades gracefully into "just search everything" rather
than confidently returning a wrong-category result.
"""
import os
import sqlite3
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = os.path.dirname(os.path.abspath(__file__))
# With 26 categories, a random guess is right ~3.8% of the time - so a flat
# threshold like 0.35 (tuned for the old 13-category model) barely ever
# triggers here even when the prediction is correct. Set the bar relative to
# that random baseline instead: needs to be at least ~3x more confident than
# chance to be trusted.
N_CATEGORIES = 26
CONFIDENCE_THRESHOLD = 3 * (1 / N_CATEGORIES)  # ~0.115 - tested against 5 threshold values,
# performance was flat (7/8 correct top category on evaluation tasks) so we use the
# cleanest theoretically-justified value (3x random-chance baseline) rather than
# an arbitrarily fitted number.

def get_conn():
    return sqlite3.connect(f"{BASE}/ai_tools.db")

def init_extra_tables():
    """Creates the tables used by ratings, saved preferences, and the
    activity log if they don't exist yet. Safe to call on every import."""
    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            stars INTEGER NOT NULL CHECK (stars BETWEEN 1 AND 5),
            comment TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            profile_name TEXT PRIMARY KEY,
            favorite_categories TEXT,
            pricing_pref TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS activity_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_text TEXT,
            predicted_category TEXT,
            confidence REAL,
            used_category_filter INTEGER,
            top_tool TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_extra_tables()

def load_catalog():
    conn = sqlite3.connect(f"{BASE}/ai_tools.db")
    df = pd.read_sql("SELECT * FROM tools", conn)
    conn.close()
    return df

# ---------------------------------------------------------------------------
# RATINGS
# ---------------------------------------------------------------------------
def add_rating(tool_name, stars, comment=""):
    conn = get_conn()
    conn.execute(
        "INSERT INTO ratings (tool_name, stars, comment) VALUES (?, ?, ?)",
        (tool_name, int(stars), comment),
    )
    conn.commit()
    conn.close()

def get_avg_ratings():
    """Returns a DataFrame of tool_name, avg_rating, num_ratings for every
    tool that has at least one rating."""
    conn = get_conn()
    df = pd.read_sql(
        "SELECT tool_name, ROUND(AVG(stars), 2) AS avg_rating, COUNT(*) AS num_ratings "
        "FROM ratings GROUP BY tool_name", conn,
    )
    conn.close()
    return df

def get_ratings_for_tool(tool_name):
    conn = get_conn()
    df = pd.read_sql(
        "SELECT stars, comment, created_at FROM ratings WHERE tool_name = ? ORDER BY created_at DESC",
        conn, params=(tool_name,),
    )
    conn.close()
    return df

# ---------------------------------------------------------------------------
# PERSONALIZATION / SAVED PREFERENCES
# ---------------------------------------------------------------------------
def save_preferences(profile_name, favorite_categories, pricing_pref):
    conn = get_conn()
    conn.execute(
        "INSERT INTO user_preferences (profile_name, favorite_categories, pricing_pref, updated_at) "
        "VALUES (?, ?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(profile_name) DO UPDATE SET favorite_categories=excluded.favorite_categories, "
        "pricing_pref=excluded.pricing_pref, updated_at=CURRENT_TIMESTAMP",
        (profile_name, ",".join(favorite_categories), pricing_pref),
    )
    conn.commit()
    conn.close()

def load_preferences(profile_name):
    conn = get_conn()
    row = conn.execute(
        "SELECT favorite_categories, pricing_pref FROM user_preferences WHERE profile_name = ?",
        (profile_name,),
    ).fetchone()
    conn.close()
    if not row:
        return None
    favorite_categories, pricing_pref = row
    return {
        "favorite_categories": [c for c in (favorite_categories or "").split(",") if c],
        "pricing_pref": pricing_pref or "All",
    }

def list_profiles():
    conn = get_conn()
    profiles = [r[0] for r in conn.execute("SELECT profile_name FROM user_preferences ORDER BY profile_name").fetchall()]
    conn.close()
    return profiles

# ---------------------------------------------------------------------------
# ANALYTICS / ACTIVITY LOG
# ---------------------------------------------------------------------------
def log_activity(task_text, predicted_category, confidence, used_category_filter, top_tool):
    conn = get_conn()
    conn.execute(
        "INSERT INTO activity_log (task_text, predicted_category, confidence, used_category_filter, top_tool) "
        "VALUES (?, ?, ?, ?, ?)",
        (task_text, predicted_category, float(confidence), int(bool(used_category_filter)), top_tool),
    )
    conn.commit()
    conn.close()

def get_activity_log():
    conn = get_conn()
    df = pd.read_sql("SELECT * FROM activity_log ORDER BY created_at DESC", conn)
    conn.close()
    return df

def load_classifier():
    model = joblib.load(f"{BASE}/category_classifier.joblib")
    vectorizer = joblib.load(f"{BASE}/classifier_vectorizer.joblib")
    return model, vectorizer

def predict_category(task_text, model, vectorizer):
    """Returns (predicted_category, confidence, error_message).
    error_message is None on success. On failure (e.g. a scikit-learn version
    mismatch between the environment the model was pickled in and the one
    running it now - the classic 'AttributeError: ... has no attribute
    multi_class') this returns (None, 0.0, <message>) instead of raising, so
    the app can fall back to searching the full catalog rather than crashing."""
    try:
        vec = vectorizer.transform([task_text])
        predicted = model.predict(vec)[0]
        probs = model.predict_proba(vec)[0]
        confidence = float(np.max(probs))
        return predicted, confidence, None
    except Exception as e:
        import sklearn
        msg = (
            f"Classifier failed to run ({type(e).__name__}: {e}). This almost always means "
            f"the installed scikit-learn version ({sklearn.__version__}) doesn't match the "
            f"version the model file was saved with. Run 'pip install -r requirements.txt' "
            f"(or 'pip install scikit-learn==1.8.0') then retrain with "
            f"'python 04_train_classifier.py' to regenerate the model for this environment. "
            f"Falling back to searching the full catalog for now."
        )
        return None, 0.0, msg

def cosine_similarity_numpy(task_vector, tool_matrix):
    dot_products = np.dot(tool_matrix, task_vector.T).flatten()
    task_norm = np.linalg.norm(task_vector)
    tool_norms = np.linalg.norm(tool_matrix, axis=1)
    denom = tool_norms * task_norm
    denom[denom == 0] = 1e-10
    return dot_products / denom

def search_fts(query, limit=15):
    """SQL full-text search - used by the 'Browse All Tools' search box.
    FTS5's default MATCH requires ALL words to appear (implicit AND), which
    is too strict for natural search phrases - "automate support tickets"
    would only match a tool containing all three exact words. We rebuild
    the query as an OR of terms instead, so any matching word counts,
    ranked by relevance (bm25) rather than treated as pass/fail.
    """
    import re
    conn = sqlite3.connect(f"{BASE}/ai_tools.db")
    terms = re.findall(r"[a-zA-Z0-9]+", query.lower())
    if not terms:
        conn.close()
        return pd.DataFrame(columns=["tool_name", "category", "pricing_tier", "best_for"])
    fts_query = " OR ".join(terms)
    try:
        results = pd.read_sql(f"""
            SELECT tools.tool_name, tools.category, tools.pricing_tier, tools.best_for, rank
            FROM tools_fts
            JOIN tools ON tools.rowid = tools_fts.rowid
            WHERE tools_fts MATCH ?
            ORDER BY rank
            LIMIT {limit}
        """, conn, params=(fts_query,))
    except Exception:
        results = pd.DataFrame(columns=["tool_name", "category", "pricing_tier", "best_for"])
    conn.close()
    return results

PRICING_ORDER = {"Free": 0, "Freemium": 1, "Paid": 2}

def recommend_tools(task_text, top_n=3, pricing_filter=None, sort_by="Relevance",
                     boost_categories=None):
    """
    sort_by: "Relevance" (default, match_score desc), "Rating" (avg rating desc),
             "Price: Low to High", "Price: High to Low", "Name (A-Z)"
    boost_categories: optional list of category names (from a saved personalization
        profile) whose tools get a small match_score boost, so a user's stated
        interests nudge ranking without overriding the text match entirely.
    """
    catalog = load_catalog()
    classifier_error = None
    try:
        model, clf_vectorizer = load_classifier()
        predicted_category, confidence, classifier_error = predict_category(task_text, model, clf_vectorizer)
    except Exception as e:
        import sklearn
        predicted_category, confidence = None, 0.0
        classifier_error = (
            f"Could not load the classifier files ({type(e).__name__}: {e}). Installed "
            f"scikit-learn is {sklearn.__version__}; make sure it matches requirements.txt "
            f"and re-run 'python 04_train_classifier.py' if needed. Falling back to searching "
            f"the full catalog for now."
        )

    used_category_filter = predicted_category is not None and confidence >= CONFIDENCE_THRESHOLD
    if used_category_filter:
        candidates = catalog[catalog["category"] == predicted_category].copy()
        if candidates.empty:
            candidates = catalog.copy()
            used_category_filter = False
    else:
        candidates = catalog.copy()  # low confidence -> search everything instead

    if pricing_filter and pricing_filter != "All":
        filtered = candidates[candidates["pricing_tier"] == pricing_filter]
        if not filtered.empty:
            candidates = filtered

    candidates["search_text"] = candidates["category"] + " " + candidates["best_for"] + " " + candidates["tags"]
    sim_vectorizer = TfidfVectorizer(stop_words="english")
    tool_matrix = sim_vectorizer.fit_transform(candidates["search_text"]).toarray()
    task_vector = sim_vectorizer.transform([task_text]).toarray()

    scores = cosine_similarity_numpy(task_vector, tool_matrix)
    candidates["match_score"] = np.round(scores, 3)

    if boost_categories:
        boost_mask = candidates["category"].isin(boost_categories)
        candidates.loc[boost_mask, "match_score"] = (candidates.loc[boost_mask, "match_score"] * 1.15 + 0.02).round(3)

    # bring in ratings so "Rating" sort and card display both have it
    ratings = get_avg_ratings()
    candidates = candidates.merge(ratings, on="tool_name", how="left")
    candidates["avg_rating"] = candidates["avg_rating"].fillna(0.0)
    candidates["num_ratings"] = candidates["num_ratings"].fillna(0).astype(int)

    candidates = candidates.sort_values("match_score", ascending=False)  # relevance pool, capped below
    pool = candidates.head(max(top_n * 4, top_n))  # keep a wider relevance-ranked pool, then re-sort for display

    if sort_by == "Rating":
        results = pool.sort_values(["avg_rating", "match_score"], ascending=[False, False]).head(top_n)
    elif sort_by == "Price: Low to High":
        pool = pool.assign(_price_rank=pool["pricing_tier"].map(PRICING_ORDER).fillna(1))
        results = pool.sort_values(["_price_rank", "match_score"], ascending=[True, False]).head(top_n)
    elif sort_by == "Price: High to Low":
        pool = pool.assign(_price_rank=pool["pricing_tier"].map(PRICING_ORDER).fillna(1))
        results = pool.sort_values(["_price_rank", "match_score"], ascending=[False, False]).head(top_n)
    elif sort_by == "Name (A-Z)":
        results = pool.sort_values("tool_name", ascending=True).head(top_n)
    else:
        results = pool.head(top_n)

    cols = ["tool_name", "category", "best_for", "pricing_tier", "official_url",
            "match_score", "avg_rating", "num_ratings"]
    return (results[cols], predicted_category, confidence, used_category_filter, classifier_error)
