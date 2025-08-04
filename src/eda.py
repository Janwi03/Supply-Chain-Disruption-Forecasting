import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def basic_inspection(df):
    print("Data Info:")
    print(df.info())
    print("\n Missing Values:")
    print(df.isnull().sum())
    print("\n First Rows:")
    print(df.head())
    print("\n Last Row:")
    print(df.tail(1))
    print("\n Target Distribution:")
    print(df['went_on_backorder'].value_counts())

def plot_kde_distributions(df):
    features = ["national_inv", "lead_time", "in_transit_qty"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, feature in enumerate(features):
        ax = axes[i]
        sns.kdeplot(
            data=df,
            x=feature,
            hue="went_on_backorder",
            common_norm=False,
            fill=True,
            palette="Set1",
            alpha=0.5,
            ax=ax
        )
        ax.set_title(f"PDF: {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Density")

    fig.suptitle("PDFs of national_inv, lead_time, in_transit_qty", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_forecast_barplots(df):
    features = ['forecast_3_month', 'forecast_6_month', 'forecast_9_month']
    titles = [
        'forecast_3_month vs went_on_backorder',
        'forecast_6_month vs went_on_backorder',
        'forecast_9_month vs went_on_backorder'
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, (feature, title) in enumerate(zip(features, titles)):
        sns.barplot(data=df, x='went_on_backorder', y=feature, ax=axes[i], palette='Set2', errorbar='ci')
        axes[i].set_title(title)

    fig.suptitle("Forecast Features vs Target")
    plt.tight_layout()
    plt.show()

def plot_violinplots(df):
    plt.figure(figsize=(8, 5))
    sns.violinplot(x='went_on_backorder', y='perf_6_month_avg', data=df)
    plt.title('Violin Plot: perf_6_month_avg vs Backorder')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.violinplot(x='went_on_backorder', y='perf_12_month_avg', data=df)
    plt.title('Violin Plot: perf_12_month_avg vs Backorder')
    plt.show()

def plot_binary_counts(df):
    cols = ['deck_risk', 'oe_constraint', 'ppap_risk', 'stop_auto_buy', 'rev_stop']
    plt.figure(figsize=(20, 5))

    for i, col in enumerate(cols):
        plt.subplot(1, len(cols), i + 1)
        sns.countplot(x=col, data=df, palette='Set2')
        plt.title(f'Count Plot: {col}')
        plt.xlabel(col)
        plt.ylabel('Count')

    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap(df):
    numeric_df = df.select_dtypes(include='float64')
    corr = numeric_df.corr(method='spearman')

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="magma",
        cbar=True, square=True, linewidths=0.5, annot_kws={"size": 8}
    )
    plt.title("Spearman Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_boxplots(df):
    numeric_df = df.select_dtypes(include='float64')
    for col in numeric_df:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot: {col}')
        plt.tight_layout()
        plt.show()

def run_full_eda(df):
    basic_inspection(df)
    plot_kde_distributions(df)
    plot_forecast_barplots(df)
    plot_violinplots(df)
    plot_binary_counts(df)
    plot_correlation_heatmap(df)
    plot_boxplots(df)