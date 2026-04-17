import os
import pandas as pd
import seaborn as sns
import matplotlib

# Use a non-interactive backend so plots can be generated in terminal-only environments.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    # Load Iris data from a CSV source using pandas
    iris_csv_url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
    df = pd.read_csv(iris_csv_url)

    print("=== Iris Dataset Overview ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    print("\n=== First 5 Rows (.head) ===")
    print(df.head())

    print("\n=== Dataset Info (.info) ===")
    df.info()

    print("\n=== Summary Statistics (.describe) ===")
    print(df.describe(include="all"))

    os.makedirs("plots", exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Scatter plot: sepal length vs sepal width, colored by species
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="sepal_length",
        y="sepal_width",
        hue="species",
        palette="Set2",
        s=80,
    )
    plt.title("Iris Scatter Plot: Sepal Length vs Sepal Width")
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.tight_layout()
    plt.savefig("plots/scatter_sepal_length_vs_width.png", dpi=150)
    plt.close()

    # Histograms for numeric feature distributions
    numeric_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numeric_cols, start=1):
        plt.subplot(2, 2, i)
        sns.histplot(data=df, x=col, kde=True, bins=20, color="#4c72b0")
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("plots/histograms_numeric_features.png", dpi=150)
    plt.close()

    # Box plots to identify potential outliers by species
    plt.figure(figsize=(10, 6))
    melted = df.melt(id_vars="species", value_vars=numeric_cols, var_name="feature", value_name="value")
    sns.boxplot(data=melted, x="feature", y="value", hue="species", palette="Set2")
    plt.title("Iris Box Plots by Feature and Species")
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig("plots/boxplots_by_species.png", dpi=150)
    plt.close()

    print("\nPlots saved in the 'plots' folder:")
    print("- plots/scatter_sepal_length_vs_width.png")
    print("- plots/histograms_numeric_features.png")
    print("- plots/boxplots_by_species.png")


if __name__ == "__main__":
    main()
