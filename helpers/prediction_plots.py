import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_prediction_score_distribution(df):
    """
    Plots the distribution of prediction probabilities to help determine where uncertainty lies.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing prediction probabilities.
    
    Returns:
    - A histogram with KDE showing the score distribution.
    """
    if "Prediction_Probability" not in df.columns:
        raise ValueError("Column 'Prediction_Probability' not found in dataframe. Ensure predictions were added.")

    plt.figure(figsize=(10, 6))
    sns.histplot(df["Prediction_Probability"], bins=50, kde=True, color="blue")
    plt.axvline(x=0.5, color="red", linestyle="--", label="Default 0.5 Threshold")
    plt.title("Distribution of Prediction Probabilities")
    plt.xlabel("Prediction Probability")
    plt.ylabel("Count")
    plt.legend()
    plt.show()


def plot_prediction_distribution(df, column_name):
    """
    Plots the distribution of a float column against the prediction probability.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing predictions
    - column_name (str): The name of the float column to plot against prediction probability
    
    Returns:
    - A histogram with KDE (Kernel Density Estimate) showing the distribution.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe.")

    if "Prediction_Probability" not in df.columns:
        raise ValueError("Column 'Prediction_Probability' not found in dataframe. Ensure predictions were added.")

    # Ensure Predicted_Label is correctly assigned (1 = Valid, 0 = Invalid)
    plt.figure(figsize=(10, 6))
    sns.histplot(df, x=column_name, hue="Predicted_Label", bins=30, kde=True, palette="coolwarm")

    plt.title(f"Distribution of {column_name} by Prediction Probability")
    plt.xlabel(column_name)
    plt.ylabel("Density")

    # Corrected Legend Order (1 = Valid, 0 = Invalid)
    plt.legend(title="Predicted Label", labels=["Valid", "Invalid"])
    plt.show()


def plot_prediction_probability_bins_with_highlight(df, column_name, highlight_bins = ["(0.45, 0.5]", "(0.5, 0.55]"]):
    """
    Plots the distribution of a float column by dividing it into bins based on the prediction probability.
    Each bin represents a range of probabilities (e.g., 0.0-0.05, 0.05-0.10, ..., 0.95-1.0).
    Highlights the bins (0.45, 0.5] and (0.5, 0.55] with a circle.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing predictions
    - column_name (str): The name of the float column to analyze
    - highlight_bins (list): ["(0.45, 0.5]", "(0.5, 0.55]"]
    
    Returns:
    - A bar plot showing the average value of the specified column in each probability bin, with row counts.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' not found in dataframe.")

    if "Prediction_Probability" not in df.columns:
        raise ValueError("Column 'Prediction_Probability' not found in dataframe. Ensure predictions were added.")

    # Create bins based on the prediction probability (0.0-0.05, 0.05-0.10, ..., 0.95-1.0)
    df["Probability_Bin"] = pd.cut(df["Prediction_Probability"], bins=np.arange(0, 1.05, 0.05), include_lowest=True)

    # Compute the mean value of the specified column per probability bin
    bin_summary = df.groupby("Probability_Bin", observed=False).agg(
        Avg_Column_Value=(column_name, "mean"),
        Count=("Prediction_Probability", "count")
    ).reset_index()

    # Convert probability bin categories to strings for plotting
    bin_summary["Probability_Bin"] = bin_summary["Probability_Bin"].astype(str)

    # Define bins to highlight
    # highlight_bins = ["(0.45, 0.5]", "(0.5, 0.55]"]

    # Plot the probability bin distribution
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x="Probability_Bin", y="Avg_Column_Value", data=bin_summary, palette="coolwarm",
                    hue="Probability_Bin", legend=False)

    # Add text annotations for row counts on top of each bar
    for p, count, bin_label in zip(ax.patches, bin_summary["Count"], bin_summary["Probability_Bin"]):
        ax.annotate(f'{count}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=12, fontweight='bold', color='black')

        # Highlight bins with a rectangle outline
        if bin_label in highlight_bins:
            rect = plt.Rectangle((p.get_x(), 0), p.get_width(), p.get_height(), 
                                 linewidth=2, edgecolor='black', facecolor='none')
            ax.add_patch(rect)

    plt.title(f"Average {column_name} per Prediction Probability Bin (with row counts)")
    plt.xlabel("Prediction Probability Bins")
    plt.ylabel(f"Average {column_name}")
    plt.xticks(rotation=90)
    plt.show()


def plot_probability_distribution(df):
    """
    Plots the distribution of prediction probabilities to check where uncertainty is highest.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing 'Prediction_Probability'.
    """
    if "Prediction_Probability" not in df.columns:
        raise ValueError("Column 'Prediction_Probability' not found in dataframe. Ensure predictions were added.")

    plt.figure(figsize=(10, 6))
    sns.histplot(df["Prediction_Probability"], bins=50, kde=True, color="blue")
    plt.axvline(x=0.5, color="red", linestyle="--", label="Default 0.5 Threshold")
    plt.title("Distribution of Prediction Probabilities")
    plt.xlabel("Prediction Probability")
    plt.ylabel("Count")
    plt.legend()
    plt.show()


def find_uncertainty_threshold(df, quantile=0.10):
    """
    Finds the range where the most uncertain samples are located based on probability distribution.

    Parameters:
    - df (pd.DataFrame): DataFrame with 'Prediction_Probability' column.
    - quantile (float): Proportion of most uncertain samples to consider (default 10%).

    Returns:
    - (low_threshold, high_threshold): Optimal range for uncertainty sampling.
    """
    sorted_probs = df["Prediction_Probability"].sort_values().values
    low_index = int(len(sorted_probs) * quantile)
    high_index = int(len(sorted_probs) * (1 - quantile))

    return sorted_probs[low_index], sorted_probs[high_index]

