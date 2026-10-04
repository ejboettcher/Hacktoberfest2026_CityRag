"""Do 311 requests justify renewing a program's budget?

Counts 311 requests per year and category, joins them to the budget,
and computes year-over-year change and cost per request.
The math happens here in pandas; the model only explains the results.
"""
import sys

import pandas as pd

# Column names vary by city. Check your CSV header and edit these.
DATE_COL = "created_date"
CATEGORY_COL = "request_type"


def requests_by_year(csv_311, categories=None):
    """Return a table of request counts per year and category."""
    df = pd.read_csv(csv_311, usecols=[DATE_COL, CATEGORY_COL])
    df["year"] = pd.to_datetime(df[DATE_COL], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    if categories:
        df = df[df[CATEGORY_COL].isin(categories)]
    return (df.groupby(["year", CATEGORY_COL]).size()
              .rename("requests").reset_index()
              .rename(columns={CATEGORY_COL: "category"}))


def demand_vs_budget(csv_311, csv_budget, categories=None):
    """Join yearly 311 demand to budget lines.

    csv_budget needs three columns: year, category, budget_usd
    (one row per program per year, typed in from the adopted budget PDFs).
    """
    counts = requests_by_year(csv_311, categories)
    budget = pd.read_csv(csv_budget)
    out = counts.merge(budget, on=["year", "category"], how="left")
    out = out.sort_values(["category", "year"])
    g = out.groupby("category")
    out["requests_change_pct"] = (g["requests"].pct_change() * 100).round(1)
    out["budget_change_pct"] = (g["budget_usd"].pct_change() * 100).round(1)
    out["usd_per_request"] = (out["budget_usd"] / out["requests"]).round(2)
    return out


def summarize(table, chat_model="gemma4:e4b"):
    """Ask the local model to explain the computed table in plain language."""
    import ollama
    prompt = (
        "Here is a table of 311 service requests and budget by year:\n\n"
        f"{table.to_string(index=False)}\n\n"
        "In plain language for a budget hearing, say whether demand rose, fell or held "
        "steady for each category, quoting only numbers that appear in the table. "
        "Note years with missing data. Do not calculate new numbers. "
        "Do not claim the budget caused or should change because of these trends; "
        "just describe them."
    )
    response = ollama.chat(model=chat_model,
                           messages=[{"role": "user", "content": prompt}],
                           options={"temperature": 0.2})
    return response["message"]["content"]


if __name__ == "__main__":
    table = demand_vs_budget(sys.argv[1], sys.argv[2], sys.argv[3:] or None)
    table.to_csv("demand_vs_budget.csv", index=False)
    print(table.to_string(index=False))
    if "--no-ai" not in sys.argv:
        print("\n" + summarize(table))
