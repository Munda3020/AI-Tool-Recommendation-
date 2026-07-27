"""
STEP 2: Load into SQLite and build TWO things:
1. The market-gap analysis (same idea as before, now across 26 categories)
2. A full-text search (FTS5) virtual table - this is a genuinely advanced
   SQL feature, not just SELECT/GROUP BY. It lets you search "video editing
   for social media" and get relevance-ranked matches across every tool's
   description, the same underlying tech real search engines use at small scale.
"""
import sqlite3
import pandas as pd

catalog = pd.read_csv("ai_tools_catalog.csv")

conn = sqlite3.connect("ai_tools.db")
catalog.to_sql("tools", conn, if_exists="replace", index=False)

# --- Build the FTS5 full-text search index ---
conn.execute("DROP TABLE IF EXISTS tools_fts")
conn.execute("""
    CREATE VIRTUAL TABLE tools_fts USING fts5(
        tool_name, category, pricing_tier, best_for, tags,
        content='tools', content_rowid='rowid'
    )
""")
conn.execute("""
    INSERT INTO tools_fts(rowid, tool_name, category, pricing_tier, best_for, tags)
    SELECT rowid, tool_name, category, pricing_tier, best_for, tags FROM tools
""")
conn.commit()

# quick test of the search index
test_query = "video editing social media"
test_results = pd.read_sql(f"""
    SELECT tools.tool_name, tools.category, tools.best_for, rank
    FROM tools_fts
    JOIN tools ON tools.rowid = tools_fts.rowid
    WHERE tools_fts MATCH '{test_query}'
    ORDER BY rank
    LIMIT 5
""", conn)
print(f"FTS5 search test for \"{test_query}\":")
print(test_results.to_string(index=False))

# --- Market-gap analysis queries ---
q1 = """
SELECT
    category,
    COUNT(*) AS tool_count,
    SUM(CASE WHEN pricing_tier = 'Free' THEN 1 ELSE 0 END) AS free_count,
    SUM(CASE WHEN pricing_tier = 'Freemium' THEN 1 ELSE 0 END) AS freemium_count,
    SUM(CASE WHEN pricing_tier = 'Paid' THEN 1 ELSE 0 END) AS paid_count
FROM tools
GROUP BY category
ORDER BY tool_count DESC;
"""
category_summary = pd.read_sql(q1, conn)
print("\n=== Category crowding (SQL) ===")
print(category_summary.to_string(index=False))

q2 = """
SELECT category, COUNT(*) AS tool_count
FROM tools
GROUP BY category
HAVING COUNT(*) <= 2
ORDER BY tool_count ASC;
"""
whitespace = pd.read_sql(q2, conn)
print(f"\n=== Whitespace categories: {len(whitespace)} of {category_summary.shape[0]} total ===")
print(whitespace.to_string(index=False))

q3 = """
SELECT
    category,
    ROUND(100.0 * SUM(CASE WHEN pricing_tier IN ('Free','Freemium') THEN 1 ELSE 0 END) / COUNT(*), 1)
        AS pct_free_or_freemium
FROM tools
GROUP BY category
ORDER BY pct_free_or_freemium DESC;
"""
pricing_mix = pd.read_sql(q3, conn)
print("\n=== % Free/Freemium by category (SQL) ===")
print(pricing_mix.head(10).to_string(index=False))

category_summary.to_csv("category_summary_sql.csv", index=False)
whitespace.to_csv("whitespace_categories_sql.csv", index=False)
pricing_mix.to_csv("pricing_mix_sql.csv", index=False)

conn.close()
print("\nSaved: ai_tools.db (with tools + tools_fts search index), 3 summary CSVs")
