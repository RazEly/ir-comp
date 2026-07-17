import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

df = pd.read_csv("data/pool.csv").drop(columns=["Employee ID"])

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

# ── 1. Numeric distributions ────────────────────────────────────────────────
num_cols = ["Age", "Monthly Income", "Years at Company", "Distance from Home",
            "Number of Promotions", "Number of Dependents", "Company Tenure"]

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Numeric Distributions", fontsize=14, fontweight="bold")
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
    axes[i].set_ylabel("Count")
    mean = df[col].mean()
    axes[i].axvline(mean, color="tomato", linestyle="--", linewidth=1.2, label=f"mean={mean:.1f}")
    axes[i].legend(fontsize=8)
axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig("experiments/plots/01_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 2. Monthly Income by Job Level & Job Role ───────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Monthly Income Distribution", fontsize=14, fontweight="bold")

level_order = ["Entry", "Mid", "Senior"]
data_by_level = [df[df["Job Level"] == lvl]["Monthly Income"] for lvl in level_order]
bp1 = ax1.boxplot(data_by_level, tick_labels=level_order, patch_artist=True,
                  medianprops={"color": "black", "linewidth": 2})
colors = ["#4e9af1", "#f1a34e", "#4ef18b"]
for patch, color in zip(bp1["boxes"], colors):
    patch.set_facecolor(color)
ax1.set_title("By Job Level")
ax1.set_ylabel("Monthly Income")
ax1.set_xlabel("Job Level")

roles = df["Job Role"].value_counts().index.tolist()
data_by_role = [df[df["Job Role"] == r]["Monthly Income"] for r in roles]
bp2 = ax2.boxplot(data_by_role, tick_labels=roles, patch_artist=True, vert=False,
                  medianprops={"color": "black", "linewidth": 2})
for patch in bp2["boxes"]:
    patch.set_facecolor("steelblue")
    patch.set_alpha(0.7)
ax2.set_title("By Job Role")
ax2.set_xlabel("Monthly Income")
plt.tight_layout()
plt.savefig("experiments/plots/02_income_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 3. Categorical satisfaction / rating distributions ───────────────────────
cat_cols = {
    "Work-Life Balance":    ["Poor", "Fair", "Good", "Excellent"],
    "Job Satisfaction":     ["Low", "Medium", "High", "Very High"],
    "Performance Rating":   ["Below Average", "Average", "High"],
    "Employee Recognition": ["Low", "Medium", "High", "Very High"],
}
palette = ["#d9534f", "#f0ad4e", "#5bc0de", "#5cb85c"]

fig, axes = plt.subplots(1, 4, figsize=(18, 5))
fig.suptitle("Categorical Rating Distributions", fontsize=14, fontweight="bold")
for ax, (col, order) in zip(axes, cat_cols.items()):
    counts = df[col].value_counts().reindex(order, fill_value=0)
    bars = ax.bar(order, counts.values,
                  color=palette[:len(order)], edgecolor="white", linewidth=0.5)
    ax.set_title(col, fontsize=10)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                f"{val:,}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("experiments/plots/03_rating_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 4. Income vs Age (scatter, colored by Job Level) ────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.suptitle("Monthly Income vs Age by Job Level", fontsize=14, fontweight="bold")
level_colors = {"Entry": "#4e9af1", "Mid": "#f1a34e", "Senior": "#4ef18b"}
for lvl, color in level_colors.items():
    mask = df["Job Level"] == lvl
    ax.scatter(df.loc[mask, "Age"], df.loc[mask, "Monthly Income"],
               c=color, alpha=0.3, s=12, label=lvl)
ax.set_xlabel("Age")
ax.set_ylabel("Monthly Income")
ax.legend(title="Job Level", markerscale=2)
plt.tight_layout()
plt.savefig("experiments/plots/04_income_vs_age.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 5. Overtime & Remote Work impact on satisfaction / performance ───────────
binary_factors = ["Overtime", "Remote Work", "Leadership Opportunities", "Innovation Opportunities"]
outcomes = ["Job Satisfaction", "Work-Life Balance", "Performance Rating"]
outcome_maps = {k: v for k, v in ordered.items() if k in outcomes}

for col, mapping in outcome_maps.items():
    df[col] = df[col].map(mapping)

fig, axes = plt.subplots(len(binary_factors), len(outcomes), figsize=(14, 14))
fig.suptitle("Binary Factors vs Outcomes (mean score)", fontsize=14, fontweight="bold")

for i, factor in enumerate(binary_factors):
    for j, outcome in enumerate(outcomes):
        ax = axes[i][j]
        group_means = df.groupby(factor)[outcome].mean()
        labels = group_means.index.tolist()
        bar_colors = ["#5cb85c" if str(l).lower() in ("yes", "1", "true") else "#d9534f"
                      for l in labels]
        bars = ax.bar([str(l) for l in labels],
                      group_means.values,
                      color=bar_colors, edgecolor="white")
        ax.set_ylim(1, group_means.values.max() * 1.2)
        ax.set_ylabel(outcome if j == 0 else "")
        ax.set_title(f"{factor}" if i == 0 else "", fontsize=9)
        for bar, val in zip(bars, group_means.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=8)
        if i == 0:
            ax.set_title(outcome, fontsize=10)
        if j == 0:
            ax.set_ylabel(factor, fontsize=9)

plt.tight_layout()
plt.savefig("experiments/plots/05_binary_factors_vs_outcomes.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 6. Promotions by Performance Rating ─────────────────────────────────────
perf_labels = {1: "Below Avg", 2: "Average", 3: "High"}
df["_perf_label"] = df["Performance Rating"].map(perf_labels)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Promotions by Performance Rating", fontsize=14, fontweight="bold")

perf_order = ["Below Avg", "Average", "High"]
data_promo = [df[df["_perf_label"] == p]["Number of Promotions"] for p in perf_order]
bp = ax1.boxplot(data_promo, tick_labels=perf_order, patch_artist=True,
                 medianprops={"color": "black", "linewidth": 2})
for patch, color in zip(bp["boxes"], ["#d9534f", "#f0ad4e", "#5cb85c"]):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
ax1.set_title("Distribution of Promotions")
ax1.set_ylabel("Number of Promotions")

means = df.groupby("_perf_label")["Number of Promotions"].mean().reindex(perf_order)
bars = ax2.bar(perf_order, means.values,
               color=["#d9534f", "#f0ad4e", "#5cb85c"], edgecolor="white")
for bar, val in zip(bars, means.values):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
             f"{val:.2f}", ha="center", va="bottom", fontsize=10)
ax2.set_title("Mean Promotions")
ax2.set_ylabel("Mean Number of Promotions")

plt.tight_layout()
plt.savefig("experiments/plots/06_promotions_by_performance.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 7. Monthly Income distribution by Job Role (violin) ─────────────────────
roles_sorted = sorted(df["Job Role"].unique())
fig, ax = plt.subplots(figsize=(15, 7))
fig.suptitle("Monthly Income Distribution by Job Role", fontsize=14, fontweight="bold")

data_by_role = [df[df["Job Role"] == r]["Monthly Income"].dropna() for r in roles_sorted]
parts = ax.violinplot(data_by_role, positions=range(len(roles_sorted)),
                      showmedians=True, showextrema=True)
for pc in parts["bodies"]:
    pc.set_facecolor("steelblue")
    pc.set_alpha(0.6)
parts["cmedians"].set_color("black")
parts["cmedians"].set_linewidth(2)

ax.set_xticks(range(len(roles_sorted)))
ax.set_xticklabels(roles_sorted, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Monthly Income")
ax.set_xlabel("Job Role")
plt.tight_layout()
plt.savefig("experiments/plots/07_income_by_role_violin.png", dpi=150, bbox_inches="tight")
plt.show()

# ── 8. Marginal distribution of EVERY column ────────────────────────────────
# Fresh read: df was mutated (ratings->ints, _perf_label) in sections 5-6.
raw = pd.read_csv("data/pool.csv").drop(columns=["Employee ID"])
all_cols = list(raw.columns)

ncols = 5
nrows = int(np.ceil(len(all_cols) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows))
fig.suptitle("Distribution of Every Column", fontsize=14, fontweight="bold")
axes = axes.flatten()

for i, col in enumerate(all_cols):
    ax = axes[i]
    s = raw[col]
    if pd.api.types.is_numeric_dtype(s):
        ax.hist(s.dropna(), bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
        ax.set_ylabel("Count")
    else:
        counts = s.value_counts()
        ax.bar(counts.index.astype(str), counts.values,
               color="mediumpurple", edgecolor="white", linewidth=0.5)
        ax.tick_params(axis="x", rotation=40, labelsize=7)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    ax.set_title(col, fontsize=9)

for j in range(len(all_cols), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("experiments/plots/08_all_columns.png", dpi=150, bbox_inches="tight")
plt.show()

print("All plots saved to experiments/plots/")
