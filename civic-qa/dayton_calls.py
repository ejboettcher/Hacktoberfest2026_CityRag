"""What are Dayton residents calling about, and what's changing?"""
import pandas as pd

df = pd.read_csv("CallsForService.csv", encoding="utf-8-sig")
df["when"] = pd.to_datetime(df["Call_Date"], format="%m/%d/%Y %I:%M:%S %p")

# Keep calls from residents; drop ones officers started themselves.
residents = df[df["Self_Initiated"].str.strip() == "no"]

# Top complaints overall
print(residents["Dispatch_Type"].value_counts().head(10))

# Compare two complete four-week windows so partial weeks don't skew trends
before = residents[(residents.when >= "2026-08-09") & (residents.when < "2026-09-06")]
after = residents[(residents.when >= "2026-09-06") & (residents.when < "2026-10-04")]
trend = pd.DataFrame({
    "aug_9_to_sep_5": before["Dispatch_Type"].value_counts(),
    "sep_6_to_oct_3": after["Dispatch_Type"].value_counts(),
}).fillna(0).astype(int)
trend["change_pct"] = ((trend.iloc[:, 1] - trend.iloc[:, 0]) / trend.iloc[:, 0] * 100).round(0)
watch = ["WELFARE CHECK", "MISSING", "ALARM COMPLAINT", "PARKING COMPLAINT",
         "LOUD NOISE COMPLAINT ETC", "ABANDONED VEHICLE", "DUMPING OR LITTERING", "TREE DOWN"]
print(trend.loc[watch])

# Is there a pothole category at all?
print(df["Dispatch_Type"].str.contains("POTHOLE|POT HOLE").any())
