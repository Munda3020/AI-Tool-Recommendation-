"""
STEP 5: Run the recommender on sample tasks, log every recommendation into
a SQL table (becomes the fact table for Power BI).
"""
import sqlite3
import pandas as pd
from recommender import recommend_tools

conn = sqlite3.connect("ai_tools.db")

test_tasks = [
    "I need to turn my notes into a pitch deck for investors",
    "find recent scientific research on this topic with sources",
    "generate a realistic voiceover for a training video",
    "help me write and debug python code for a data pipeline",
    "translate this contract into another language",
    "automate customer support ticket responses",
    "check my website's SEO ranking against competitors",
    "turn my photo into a talking avatar video",
]

log_rows = []
for task in test_tasks:
    results, predicted_category, confidence, used_filter = recommend_tools(task, top_n=3)
    print(f"\nTASK: \"{task}\"")
    print(f"  Predicted category: {predicted_category} (confidence: {confidence:.1%}, "
          f"{'used category filter' if used_filter else 'searched full catalog'})")
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        print(f"  {rank}. {row['tool_name']} ({row['category']}, {row['pricing_tier']}) | score: {row['match_score']}")
        log_rows.append({
            "task": task, "rank": rank, "tool_name": row["tool_name"],
            "category": row["category"], "match_score": row["match_score"],
            "predicted_category": predicted_category, "classifier_confidence": round(confidence, 3),
            "used_category_filter": used_filter
        })

log_df = pd.DataFrame(log_rows)
log_df.to_sql("recommendation_log", conn, if_exists="replace", index=False)
log_df.to_csv("recommendation_log.csv", index=False)
print(f"\nSaved {len(log_df)} logged recommendations to SQL table + CSV")
conn.close()
