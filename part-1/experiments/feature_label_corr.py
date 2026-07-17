"""
Correlation of each feature with the label (Left=1).
Uses the leaked full pool labels (diagnostic only, not submittable).
Run from part-1/: python experiments/feature_label_corr.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
from utils import ID_COLUMN, _load_pool_labels, load_pool

OUT = Path(__file__).resolve().parent / "plots" / "08_feature_label_correlation.png"

df = load_pool()
labels = _load_pool_labels()
df["Left"] = df[ID_COLUMN].astype(str).map(labels).astype(float)
df = df.drop(columns=[ID_COLUMN])

# ordinal encodings (same mappings as corr_matrix.py)
ordered = {
    "Work-Life Balance":    {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4},
    "Job Satisfaction":     {"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
    "Performance Rating":   {"Below Average": 1, "Average": 2, "High": 3},
    "Education Level":      {"High School": 1, "Associate Degree": 2, "Bachelor's Degree": 3, "Master's Degree": 4, "PhD": 5},
    "Job Level":            {"Entry": 1, "Mid": 2, "Senior": 3},
    "Company Size":         {"Small": 1, "Medium": 2, "Large": 3},
    "Company Reputation":   {"Poor": 1, "Fair": 2, "Good": 3, "Excellent": 4},
    "Employee Recognition": {"Low": 1, "Medium": 2, "High": 3, "Very High": 4},
}
for col, mapping in ordered.items():
    if col in df.columns:
        df[col] = df[col].map(mapping)

binary = ["Gender", "Overtime", "Remote Work", "Leadership Opportunities", "Innovation Opportunities"]
for col in binary:
    if col in df.columns:
        df[col] = (df[col] == sorted(df[col].dropna().unique())[-1]).astype(int)

for col in ["Job Role", "Marital Status"]:
    if col in df.columns:
        df[col] = pd.Categorical(df[col]).codes

# correlation of each feature with Left
corr = df.corr(numeric_only=True)["Left"].drop("Left").sort_values()

fig, ax = plt.subplots(figsize=(9, max(5, 0.4 * len(corr))))
colors = ["#c0392b" if v > 0 else "#2471a3" for v in corr.values]
ax.barh(corr.index, corr.values, color=colors)
ax.axvline(0, color="black", lw=0.8)
for i, v in enumerate(corr.values):
    ax.text(v + (0.005 if v >= 0 else -0.005), i, f"{v:.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=8)
ax.set_xlabel("Pearson correlation with Left (=1)")
ax.set_title("Feature correlation with attrition label\n(red = higher Left risk, blue = retention)")
pad = max(abs(corr.min()), abs(corr.max())) * 0.25
ax.set_xlim(corr.min() - pad, corr.max() + pad)
plt.tight_layout()
OUT.parent.mkdir(exist_ok=True)
plt.savefig(OUT, dpi=130)
print(f"saved {OUT}")
print(corr.round(3).to_string())
