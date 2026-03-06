# Customer Personality Analysis
# Exploratory data analysis, regression, classification, and clustering
# on a 2,240-record marketing dataset to identify customer segments
# and predict campaign response behavior.

import os
import matplotlib
if os.environ.get("MPLBACKEND") is None and not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import (
    GridSearchCV, cross_val_predict, train_test_split
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, make_scorer,
    accuracy_score, confusion_matrix, classification_report
)

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(filename):
    plt.savefig(os.path.join(FIGURES_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Data Loading & Cleaning
# ============================================================

DATA_URL = (
    "http://www.csc.calpoly.edu/~dekhtyar/301-Winter2024/data/marketing_campaign.csv"
)

df = pd.read_csv(DATA_URL, delimiter="\t").set_index("ID")

print(f"Dataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# ============================================================
# Part 1 — Exploratory Analysis & Feature Engineering
# ============================================================

def plot_histogram(data, column, title, xlabel, ylabel="Count", filename=None):
    plt.figure(figsize=(10, 6))
    plt.hist(data[column], bins=10, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if filename:
        save_fig(filename)
    else:
        plt.close()


# --- Birth year distribution ---
birth_years = df["Year_Birth"].values.reshape(-1, 1)
scaler = StandardScaler().fit(birth_years)
print(f"Birth year — mean: {scaler.mean_[0]:.1f}, std: {np.sqrt(scaler.var_[0]):.1f}")

plot_histogram(df, "Year_Birth", "Distribution of Customer Birth Year",
               "Year of Birth", filename="01_birth_year_distribution.png")

# --- Income distribution (remove outliers > 130 000) ---
plot_histogram(df, "Income", "Income Distribution (raw)",
               "Income", filename="02_income_raw.png")

df_clean = df[df["Income"] < 130_000].copy()
plot_histogram(df_clean, "Income", "Income Distribution (outliers removed)",
               "Income", filename="03_income_cleaned.png")

# --- Income bins based on quartiles ---
quantiles = df["Income"].quantile([0.25, 0.50, 0.75])

def categorize_income(income):
    if income < quantiles[0.25]:
        return "Low"
    elif income < quantiles[0.50]:
        return "Lower-mid"
    elif income < quantiles[0.75]:
        return "Upper-mid"
    return "Upper"

df["IncomeBin"] = df["Income"].apply(categorize_income)

# --- Campaign acceptance flag ---
campaign_cols = [f"AcceptedCmp{i}" for i in range(1, 6)]
df["AcceptedCmp"] = (df[campaign_cols].sum(axis=1) > 0).astype(int)

campaign_rates = (df[campaign_cols].mean() * 100).round(2)
print(f"Campaign acceptance rates (%):\n{campaign_rates}")

accepted_pct = (df["AcceptedCmp"].mean() * 100).round(2)
print(f"Customers who accepted at least one campaign: {accepted_pct}%")

plt.figure(figsize=(6, 4))
plt.bar(["Accepted ≥1 Offer"], [accepted_pct], color="green")
plt.ylim(0, 100)
plt.ylabel("Percentage (%)")
plt.title("Customers Who Accepted at Least One Campaign Offer")
save_fig("04_campaign_acceptance.png")

# --- Complaints vs campaign acceptance ---
crosstab = pd.crosstab(df["AcceptedCmp"], df["Complain"], margins=True)
print(crosstab)

# --- Total purchases ---
df["Purchases"] = (
    df["NumWebPurchases"] + df["NumCatalogPurchases"] + df["NumStorePurchases"]
)
plot_histogram(df, "Purchases", "Distribution of Total Purchases",
               "Purchases", filename="05_purchases_distribution.png")

# --- Combined scatter: birth year × income × marital status × purchases ---
marital_map = {
    "Married": "Together/Married", "Together": "Together/Married",
    "Alone": "Alone/Widow",       "Widow": "Alone/Widow",
    "Single": "Single",           "Divorced": "Divorced",
    "Absurd": "YOLO/Absurd",      "YOLO": "YOLO/Absurd",
}
plot_df = df[df["Income"] <= 130_000].copy()
plot_df["Marital_Status"] = plot_df["Marital_Status"].map(marital_map)

plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=plot_df, x="Year_Birth", y="Income",
    hue="Marital_Status", size="Purchases",
    sizes=(50, 300), palette="deep", alpha=0.7,
)
plt.title("Birth Year vs Income (by Marital Status & Purchases)")
plt.xlabel("Year of Birth")
plt.ylabel("Income")
plt.grid(True)
save_fig("06_multidim_scatter.png")


# ============================================================
# Part 2 — Regression
# ============================================================

df_reg = df.dropna(subset=["Income"])
df_reg = df_reg[df_reg["Income"] <= 130_000]

# --- Simple linear regression: birth year → income ---
X_birth = df_reg["Year_Birth"].values.reshape(-1, 1)
y_income = df_reg["Income"].values

lr_simple = LinearRegression().fit(X_birth, y_income)
print(f"Birth year → Income slope: {lr_simple.coef_[0]:.2f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_reg, x="Year_Birth", y="Income", label="Data")
plt.plot(df_reg["Year_Birth"], lr_simple.predict(X_birth), color="red", label="Regression Line")
plt.title("Linear Regression: Birth Year vs Income")
plt.xlabel("Year of Birth")
plt.ylabel("Income")
plt.legend()
plt.grid(True)
save_fig("07_birth_year_regression.png")


def print_regression_metrics(y_true, y_pred, label):
    mse = mean_squared_error(y_true, y_pred)
    print(f"[{label}]  MSE: {mse:.2f}  RMSE: {np.sqrt(mse):.2f}  "
          f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")


# --- Multi-feature regression: product spending → income ---
spending_cols = [
    "MntWines", "MntFruits", "MntMeatProducts",
    "MntFishProducts", "MntSweetProducts", "MntGoldProds",
]

train_df = df[:2000].dropna()
valid_df = df[2000:].dropna()

lr_spending = LinearRegression()
lr_spending.fit(train_df[spending_cols], train_df["Income"])

print_regression_metrics(train_df["Income"], lr_spending.predict(train_df[spending_cols]), "Train")
print_regression_metrics(valid_df["Income"], lr_spending.predict(valid_df[spending_cols]), "Valid")

# --- Extended model: spending + demographics → income ---
cat_features = ["Education", "Marital_Status"]
num_features = spending_cols + ["Year_Birth"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_features),
])
pipeline = Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])
pipeline.fit(train_df[num_features + cat_features], train_df["Income"])

y_train_pred = pipeline.predict(train_df[num_features + cat_features])
y_valid_pred = pipeline.predict(valid_df[num_features + cat_features])

print_regression_metrics(train_df["Income"], y_train_pred, "Train (extended)")
print_regression_metrics(valid_df["Income"], y_valid_pred, "Valid (extended)")

plt.figure(figsize=(10, 6))
plt.scatter(valid_df["Income"], y_valid_pred, alpha=0.5, color="blue")
plt.plot(
    [valid_df["Income"].min(), valid_df["Income"].max()],
    [valid_df["Income"].min(), valid_df["Income"].max()],
    color="red", linestyle="--", label="y = x",
)
plt.title("Predicted vs Actual Income (Validation Set)")
plt.xlabel("Actual Income")
plt.ylabel("Predicted Income")
plt.legend()
plt.grid(True)
save_fig("08_predicted_vs_actual.png")


# ============================================================
# Part 3 — Classification
# ============================================================

# --- Logistic regression: campaign history → last-campaign response ---
X_cls = df[campaign_cols + ["NumDealsPurchases"]]
y_cls = df["Response"]

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

print(f"Logistic Regression — Train acc: {accuracy_score(y_train, log_reg.predict(X_train)):.4f}")
print(f"Logistic Regression — Test  acc: {accuracy_score(y_test, log_reg.predict(X_test)):.4f}")
print("Train confusion matrix:\n", confusion_matrix(y_train, log_reg.predict(X_train)))
print("Test  confusion matrix:\n", confusion_matrix(y_test, log_reg.predict(X_test)))

# --- KNN with grid search: spending + purchase channels → response ---
knn_features = spending_cols + [
    "NumWebPurchases", "NumCatalogPurchases",
    "NumStorePurchases", "NumWebVisitsMonth",
]

param_grid = {
    "n_neighbors": range(1, 12),
    "metric": ["euclidean", "manhattan", "minkowski"],
}

grid_search = GridSearchCV(
    KNeighborsClassifier(), param_grid,
    cv=5, scoring=make_scorer(accuracy_score), return_train_score=True,
)
grid_search.fit(df[knn_features], df["Response"])

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV accuracy: {grid_search.best_score_:.4f}")

results = pd.DataFrame(grid_search.cv_results_)
pivot = results.pivot(
    index="param_n_neighbors", columns="param_metric", values="mean_test_score"
)
pivot.plot(kind="bar", figsize=(12, 8))
plt.title("KNN Cross-Validated Accuracy by Hyperparameters")
plt.xlabel("Number of Neighbors")
plt.ylabel("Mean Test Accuracy")
plt.grid(True)
save_fig("09_knn_gridsearch.png")

best_knn = grid_search.best_estimator_
y_cv_pred = cross_val_predict(best_knn, df[knn_features], df["Response"], cv=5)
print("Best KNN — Confusion Matrix:\n", confusion_matrix(df["Response"], y_cv_pred))
print("Best KNN — Classification Report:\n", classification_report(df["Response"], y_cv_pred))


# ============================================================
# Part 4 — Clustering
# ============================================================

df_clust = df[df["Income"] <= 130_000].copy()
X_mnt = df_clust[spending_cols]

# Without scaling
km_raw = KMeans(n_clusters=4, init="random", n_init=1, random_state=100)
df_clust["Cluster_Raw"] = km_raw.fit_predict(X_mnt)

# With StandardScaler
X_mnt_scaled = StandardScaler().fit_transform(X_mnt)
km_scaled = KMeans(n_clusters=4, random_state=42)
df_clust["Cluster_Scaled"] = km_scaled.fit_predict(X_mnt_scaled)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, col, title in zip(
    axes,
    ["Cluster_Raw", "Cluster_Scaled"],
    ["K-Means (unscaled)", "K-Means (StandardScaler)"],
):
    scatter = ax.scatter(
        df_clust["Year_Birth"], df_clust["Income"],
        c=df_clust[col], cmap="viridis", alpha=0.6,
    )
    ax.set_title(title)
    ax.set_xlabel("Birth Year")
    ax.set_ylabel("Income")
    fig.colorbar(scatter, ax=ax, label="Cluster")

plt.tight_layout()
save_fig("10_kmeans_clustering.png")

print("Cluster centroids (scaled model):")
centroids_df = pd.DataFrame(
    StandardScaler().fit(X_mnt).inverse_transform(km_scaled.cluster_centers_),
    columns=spending_cols,
)
print(centroids_df.round(1))
