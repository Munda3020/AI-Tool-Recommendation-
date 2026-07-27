# 🤖 AI Tool Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![SQLite](https://img.shields.io/badge/SQLite-07405E?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)

An advanced market intelligence platform and recommendation engine analyzing **107 AI tools** across **26 categories**. Featuring an **82.7% accurate ML classifier**, custom similarity engines, and a comprehensive 6-page interactive dashboard.

---

## 🚀 Quick Start

Get the platform running in three simple commands:

```bash
# Install dependencies
pip install -r requirements.txt

# Rebuild data pipeline (Catalog -> SQL -> ML Training -> Exports)
python run_pipeline.py

# Launch the interactive dashboard
streamlit run app.py
```

> **Note:** `run_pipeline.py` is the backbone of the project, automating the entire data lifecycle from scratch.

---

## 📊 Market Insights & Performance

### 🧠 Model Performance Comparison
We benchmarked multiple architectures to find the most robust classifier for tool categorization. The **Tuned Logistic Regression** emerged as the winner through rigorous hyperparameter optimization.

![Model Performance](assets/model_performance.png)

### 📈 Market Concentration
Our analysis reveals significant crowding in **Video Generation** and **Coding** sectors, while identifying **9 whitespace categories** ripe for innovation.

![Category Distribution](assets/category_distribution.png)

### 💰 Pricing Strategy Analysis
The AI market is dominated by **Freemium** models (71%), while certain professional sectors like SEO and Customer Support maintain a strict **Paid-only** barrier.

![Pricing Mix](assets/pricing_mix.png)

---

## 🛠️ Technical Architecture

The platform utilizes a sophisticated **Two-Stage Recommender System**:

1.  **ML Classification Layer**: A Logistic Regression model (word + character n-grams) predicts the task category with high precision.
2.  **Semantic Ranking Layer**: A custom NumPy-based cosine similarity engine ranks specific tools within the predicted category.
3.  **Intelligent Fallback**: If classification confidence drops below 11.5%, the system automatically switches to a global catalog search to ensure relevant results.

### Tech Stack Highlights
*   **Data Science**: `pandas`, `numpy`, `scikit-learn` (GridSearchCV, TF-IDF)
*   **Database**: `SQLite` with `FTS5` for high-performance full-text search.
*   **Visualization**: `Streamlit`, `Plotly`, and `Seaborn`.
*   **BI Integration**: Automated star-schema exports for **Power BI** analysis.

---

## 📂 Project Structure

| Component | Responsibility |
| :--- | :--- |
| `01_build_catalog.py` | Catalog construction with 107 verified tool URLs. |
| `02_sql_market_analysis.py` | Market analysis and FTS5 search index generation. |
| `03_generate_training_data.py` | Synthetic and real-world training data expansion (488 examples). |
| `04_train_classifier.py` | Model benchmarking and GridSearchCV hyperparameter tuning. |
| `05_test_recommender.py` | Validation suite for recommendation accuracy. |
| `06_export_for_powerbi.py` | Data transformation for Business Intelligence tools. |
| `app.py` | The main interactive 6-page Streamlit dashboard. |

---

## 🎯 Key Findings (2026)
*   **107 Tools / 26 Categories**: Comprehensive coverage of the current AI landscape.
*   **Accuracy Leap**: Model accuracy improved from **38.6% → 82.7%** through character n-gram integration and data expansion.
*   **Market Gaps**: Identified underserved niches in *Legal Tech*, *3D Spatial Design*, and *Enterprise Knowledge*.

---

## 🔮 Future Roadmap
*   Transition from TF-IDF to **Transformer-based Sentence Embeddings**.
*   Expand training dataset to **1,300+ labeled examples**.
*   Implement real-time user feedback loops for continuous model refinement.

---
*Created as a showcase for AI Market Intelligence & Machine Learning Engineering.*
