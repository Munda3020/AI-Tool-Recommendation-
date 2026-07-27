"""
STEP 6: Export clean Power BI-ready tables (star schema: dimension + fact).
"""
import sqlite3
import pandas as pd

conn = sqlite3.connect("ai_tools.db")

tools = pd.read_sql("SELECT * FROM tools", conn)
tools.to_csv("powerbi_tools.csv", index=False)

log = pd.read_sql("SELECT * FROM recommendation_log", conn)
log.to_csv("powerbi_recommendation_log.csv", index=False)

conn.close()
print("Power BI-ready files saved: powerbi_tools.csv, powerbi_recommendation_log.csv")
