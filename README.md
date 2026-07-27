# AI Tool Intelligence Platform

A researched catalog of 107 real AI tools across 26 categories (2026), with an
82.7%-accurate ML classifier, a hand-implemented numpy similarity engine,
SQL-driven market analysis with full-text search, clickable links to every
tool, and a professionally styled 6-page dashboard.

## Quick Start (tested end-to-end, 3 commands)
```
pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```
`run_pipeline.py` rebuilds every data file, database, and trained model from
scratch, in order. The app opens at `http://localhost:8501`.

*(Optional: deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud)
for a live link to put on your resume.)*

## Business Question
With 100+ active AI tools spanning writing, coding, video, automation, legal,
and a dozen other categories, choosing the right one is genuinely hard. Can a
system recommend the right tool for a task, link straight to it, let people
search the market by meaning rather than exact keywords, and reveal where the
tool market itself is crowded vs. underserved?

## Tech Stack
Python (pandas, numpy) · scikit-learn (TF-IDF word+character n-grams, 4
benchmarked classifiers, GridSearchCV) · SQL (SQLite + FTS5 full-text search)
· Streamlit + Plotly (6-page dashboard, custom theme) · Power BI-ready exports

## Accuracy: 82.7% Held-Out / 75.2% CV Mean (up from an initial 38.6%)
Reached through evidence at every step, not by reducing the number of
categories:
1. Training data expanded from 176 → 488 examples across 26 categories
2. Combined **word n-grams (1,2) + character n-grams (3,5)** — character
   n-grams catch morphological variants like "debug"/"debugging" that
   word-only TF-IDF misses entirely
3. **Benchmarked 4 approaches** (Naive Bayes, Logistic Regression, Linear
   SVM, a soft-voting ensemble of the two) via 3-fold cross-validation, then
   **hyperparameter-tuned the winner** (Logistic Regression, `C=3`) via
   GridSearchCV instead of using library defaults
4. Both the held-out accuracy (82.7%) and the more conservative CV mean
   (75.2%) are reported — they differ because some categories have as few
   as 3 test examples, so the single held-out split carries real variance.
   Reporting only the higher number would be misleading.

## A Rendering Bug Also Fixed (worth knowing as a Streamlit gotcha)
An earlier version of this app displayed tool cards as raw HTML text (e.g.
literally showing `<p class="tool-name">Framer AI</p>` on screen) instead of
rendering them. Cause: the HTML was written as an indented multi-line Python
string to look tidy in the source code, but Markdown treats 4+ leading spaces
as a literal code block — so the "pretty" indentation silently broke
rendering. Fixed with `textwrap.dedent()` before passing to `st.markdown()`.
Verified via Streamlit's `AppTest` framework by asserting the rendered HTML's
first line has zero leading whitespace, not just "the page loads."

## What's In the Dashboard — 6 Pages
1. **Overview** — KPIs and catalog-wide charts
2. **Tool Recommender** — describe a task, get ranked recommendations with direct links
3. **Browse & Search** — SQL full-text search (FTS5, bm25 ranking) with filters
4. **Market Gap Dashboard** — category crowding, whitespace, pricing mix
5. **Compare Tools** — side-by-side cards for any 2-4 tools
6. **Model Performance** — the 4-model comparison, confusion matrix, honest accuracy numbers

## UI
Custom dark theme with a consistent color system across CSS and every Plotly
chart. Every tool shows its real logo and a working link button straight to
its actual website (all 107 URLs verified).

## Architecture — Two-Stage Recommender With a Fallback
1. **ML classifier (learned)**: tuned Logistic Regression on word+char
   n-gram features predicts the task's category (now typically 40-90%
   confidence on real tasks, up from 5-30% before the accuracy work).
2. **Similarity ranking (calculated)**: numpy-implemented cosine similarity
   (`np.dot`, `np.linalg.norm`) ranks tools within that category.
3. **Confidence-based fallback**: below ~11.5% confidence (3x random chance
   for 26 classes), the recommender searches the entire catalog instead of
   trusting a shaky prediction. With the improved classifier this triggers
   far less often — but still protects against genuinely ambiguous tasks.

## Key Findings
- **107 tools across 26 categories** — Video Generation and Coding &
  Development are the most crowded (10 tools each)
- **9 whitespace categories** (≤2 tools) — Translation, Legal & Compliance,
  Avatar & Digital Human, 3D & Spatial Design, Email & Sales, Notes &
  Knowledge Management, Enterprise Knowledge, Creative Writing, Website Building
- **SEO and Customer Support are 100% paid** — free-tier competition there is
  essentially absent

## Building the Power BI Dashboard
1. Get Data → Text/CSV → load `powerbi_tools.csv` and `powerbi_recommendation_log.csv`
2. Relationship: `powerbi_tools[tool_name]` → `powerbi_recommendation_log[tool_name]`
3. DAX measures:
```
Total Tools = DISTINCTCOUNT(powerbi_tools[tool_name])
Whitespace Categories = CALCULATE(DISTINCTCOUNT(powerbi_tools[category]), FILTER(VALUES(powerbi_tools[category]), CALCULATE(COUNTROWS(powerbi_tools)) <= 2))
% Free or Freemium = DIVIDE(CALCULATE(COUNTROWS(powerbi_tools), powerbi_tools[pricing_tier]<>"Paid"), COUNTROWS(powerbi_tools))
```

## What Each File Does
| File | What it does |
|---|---|
| `run_pipeline.py` | **Run this first** — rebuilds everything in order |
| `01_build_catalog.py` | Builds the 107-tool catalog with verified URLs (pandas) |
| `02_sql_market_analysis.py` | SQL market analysis + builds the FTS5 search index |
| `03_generate_training_data.py` | 488 labeled examples across 26 categories |
| `04_train_classifier.py` | Benchmarks 4 approaches + GridSearchCV tuning, deploys the winner |
| `05_test_recommender.py` | Runs sample recommendations, logs to SQL |
| `06_export_for_powerbi.py` | Exports star-schema tables for Power BI |
| `recommender.py` | Shared logic: classifier + numpy similarity + fallback + FTS search |
| `app.py` | **The professionally styled 6-page Streamlit dashboard** |

## What I'd Do With More Time
- Collect 50+ labeled examples per category (1,300+ total) and re-measure accuracy
- Replace TF-IDF with sentence embeddings for genuine semantic matching
- Deploy publicly and log real user clicks to measure recommendation quality honestly
