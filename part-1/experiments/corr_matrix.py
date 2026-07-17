import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("pool.csv").drop(columns=["Employee ID"])

# ordered
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
    df[col] = df[col].map(mapping)

# binary
binary = ["Gender", "Overtime", "Remote Work", "Leadership Opportunities", "Innovation Opportunities"]
for col in binary:
    df[col] = (df[col] == df[col].unique()[1]).astype(int)

# nominal (no order) — label encode
for col in ["Job Role", "Marital Status"]:
    df[col] = pd.Categorical(df[col]).codes

corr = df.corr()

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax)
ax.set_xticks(range(len(corr)))
ax.set_yticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(corr.columns, fontsize=8)
for i in range(len(corr)):
    for j in range(len(corr)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6)
plt.tight_layout()
plt.show()
