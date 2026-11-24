import hashlib
import pathlib

import pandas as pd

p = pathlib.Path(__file__).parent / "data" / "Telco-Customer-Churn.csv"
raw = p.read_bytes()
print("bytes", len(raw), "sha256", hashlib.sha256(raw).hexdigest())
df = pd.read_csv(p)
print("shape", df.shape)
print("columns", list(df.columns))
print("churn_rate", round((df["Churn"] == "Yes").mean(), 4))
print("dup_ids", int(df["customerID"].duplicated().sum()))
print("missing", df.isna().sum().to_dict())
tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
print("totalcharges_blank", int(tc.isna().sum()), "blank_with_zero_tenure",
      int((tc.isna() & (df["tenure"] == 0)).sum()))
for col in ["gender", "Contract", "InternetService", "PaymentMethod", "PaperlessBilling"]:
    print(col, df[col].nunique(), sorted(df[col].unique())[:6])
print("tenure_range", int(df["tenure"].min()), int(df["tenure"].max()))
print("monthly", round(df["MonthlyCharges"].min(), 2), round(df["MonthlyCharges"].max(), 2))
print(df["Churn"].value_counts().to_dict())
