import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. LOAD CLEANED DATA
# ==========================================

df = pd.read_csv("data/crowdfunding_cleaned.csv")

print("Dataset Loaded Successfully!")
print("Shape:", df.shape)


# ==========================================
# 2. BASIC INFORMATION
# ==========================================

print("\n========== BASIC INFORMATION ==========")

print("\nColumns:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)


# ==========================================
# 3. SUCCESS ANALYSIS
# ==========================================

print("\n========== SUCCESS ANALYSIS ==========")

success_count = df["Success"].value_counts()

print(success_count)

print("\nSuccess Percentage:")
print(df["Success"].value_counts(normalize=True) * 100)


# ==========================================
# 4. CATEGORY ANALYSIS
# ==========================================

print("\n========== CATEGORY ANALYSIS ==========")

category_success = (
    df.groupby("Category")["Success"]
    .mean()
    .sort_values(ascending=False)
    * 100
)

print(category_success)


# ==========================================
# 5. TOP 10 CATEGORIES
# ==========================================

print("\n========== TOP 10 CATEGORIES ==========")

print(category_success.head(10))


# ==========================================
# 6. GOAL AMOUNT ANALYSIS
# ==========================================

print("\n========== GOAL ANALYSIS ==========")

print(
    df.groupby("Success")["Goal_Amount"]
    .mean()
)


# ==========================================
# 7. CAMPAIGN DURATION ANALYSIS
# ==========================================

print("\n========== DURATION ANALYSIS ==========")

print(
    df.groupby("Success")["Duration_Days"]
    .mean()
)


# ==========================================
# 8. BACKER ANALYSIS
# ==========================================

print("\n========== BACKER ANALYSIS ==========")

print(
    df.groupby("Success")["Backer_Count"]
    .mean()
)


# ==========================================
# 9. FUNDING RATIO ANALYSIS
# ==========================================

print("\n========== FUNDING RATIO ==========")

print(
    df.groupby("Success")["Funding_Ratio"]
    .mean()
)


# ==========================================
# 10. LAUNCH YEAR ANALYSIS
# ==========================================

print("\n========== YEAR ANALYSIS ==========")

year_success = (
    df.groupby("Launch_Year")["Success"]
    .mean()
    * 100
)

print(year_success)


# ==========================================
# 11. VISUALIZATION 1
# SUCCESS VS FAILURE
# ==========================================

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="Success"
)

plt.title("Successful vs Failed Campaigns")
plt.xlabel("Campaign Status")
plt.ylabel("Number of Campaigns")

plt.xticks(
    [0, 1],
    ["Failed", "Successful"]
)

plt.tight_layout()
plt.show()


# ==========================================
# 12. VISUALIZATION 2
# SUCCESS BY CATEGORY
# ==========================================

plt.figure(figsize=(12, 6))

category_success.head(10).sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Categories by Success Rate")
plt.xlabel("Success Rate (%)")
plt.ylabel("Category")

plt.tight_layout()
plt.show()


# ==========================================
# 13. VISUALIZATION 3
# GOAL VS SUCCESS
# ==========================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Success",
    y="Goal_Amount"
)

plt.ylim(
    0,
    df["Goal_Amount"].quantile(0.95)
)

plt.title("Funding Goal vs Campaign Success")
plt.xlabel("Campaign Status")
plt.ylabel("Funding Goal")

plt.xticks(
    [0, 1],
    ["Failed", "Successful"]
)

plt.tight_layout()
plt.show()


# ==========================================
# 14. VISUALIZATION 4
# DURATION VS SUCCESS
# ==========================================

plt.figure(figsize=(8, 6))

sns.boxplot(
    data=df,
    x="Success",
    y="Duration_Days"
)

plt.title("Campaign Duration vs Success")
plt.xlabel("Campaign Status")
plt.ylabel("Duration (Days)")

plt.xticks(
    [0, 1],
    ["Failed", "Successful"]
)

plt.tight_layout()
plt.show()


# ==========================================
# 15. VISUALIZATION 5
# SUCCESS BY YEAR
# ==========================================

plt.figure(figsize=(10, 5))

year_success.plot(
    marker="o"
)

plt.title("Campaign Success Rate by Launch Year")
plt.xlabel("Launch Year")
plt.ylabel("Success Rate (%)")
plt.grid(True)

plt.tight_layout()
plt.show()


# ==========================================
# 16. VISUALIZATION 6
# CATEGORY DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 6))

df["Category"].value_counts().head(10).plot(
    kind="bar"
)

plt.title("Top 10 Campaign Categories")
plt.xlabel("Category")
plt.ylabel("Number of Campaigns")

plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


print("\n========================================")
print("EDA COMPLETED SUCCESSFULLY!")
print("========================================")