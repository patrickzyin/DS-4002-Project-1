#!/usr/bin/env python3
"""
Statistical Analysis: Crisis vs Non-Crisis State of the Union Speeches

This script:
1. Loads sentiment analysis results
2. Compares crisis vs non-crisis speeches
3. Performs two-sample t-test
4. Creates visualizations (separate images)
5. Saves summary report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os


# Load the sentiment analysis results
def load_data():
    data_path = os.path.join('..', 'OUTPUTS', 'sentiment_analysis_results.csv')
    print("Loading sentiment analysis results...")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} speeches from {df['year'].min()} to {df['year'].max()}\n")
    return df


# Prepare data for analysis
def prepare_data(df):
    df['is_crisis'] = df['is_crisis'].astype(bool)
    
    # Calculate ratio of positive to negative sentences
    df['positive_negative_ratio'] = df['positive_count'] / df['negative_count']
    
    # Separate crisis and non-crisis speeches
    crisis = df[df['is_crisis'] == True].copy()
    non_crisis = df[df['is_crisis'] == False].copy()
    
    print(f"Crisis speeches: {len(crisis)}")
    print(f"Non-crisis speeches: {len(non_crisis)}\n")
    
    return df, crisis, non_crisis


# Calculate and display descriptive statistics
def descriptive_statistics(crisis, non_crisis):
    print("Descriptive Statistics\n")
    
    print("Crisis Speeches:")
    print(f"  Count: {len(crisis)}")
    print(f"  Mean positive %: {crisis['positive_percentage'].mean():.2f}%")
    print(f"  Std dev: {crisis['positive_percentage'].std():.2f}%")
    print(f"  Median positive %: {crisis['positive_percentage'].median():.2f}%")
    print(f"  Min: {crisis['positive_percentage'].min():.2f}%")
    print(f"  Max: {crisis['positive_percentage'].max():.2f}%")
    
    print("\nNon-Crisis Speeches:")
    print(f"  Count: {len(non_crisis)}")
    print(f"  Mean positive %: {non_crisis['positive_percentage'].mean():.2f}%")
    print(f"  Std dev: {non_crisis['positive_percentage'].std():.2f}%")
    print(f"  Median positive %: {non_crisis['positive_percentage'].median():.2f}%")
    print(f"  Min: {non_crisis['positive_percentage'].min():.2f}%")
    print(f"  Max: {non_crisis['positive_percentage'].max():.2f}%")
    
    print("\nDifference:")
    diff = crisis['positive_percentage'].mean() - non_crisis['positive_percentage'].mean()
    print(f"  Crisis - Non-crisis: {diff:.2f} percentage points\n")


# Perform two-sample t-test
def perform_ttest(crisis, non_crisis):
    print("Two-Sample T-Test\n")
    
    crisis_ratios = crisis['positive_negative_ratio'].dropna()
    non_crisis_ratios = non_crisis['positive_negative_ratio'].dropna()
    
    print("Using positive/negative ratio as the metric")
    print(f"Crisis speeches: n={len(crisis_ratios)}, mean ratio={crisis_ratios.mean():.3f}")
    print(f"Non-crisis speeches: n={len(non_crisis_ratios)}, mean ratio={non_crisis_ratios.mean():.3f}")
    
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(crisis_ratios, non_crisis_ratios)
    
    print("\nResults:")
    print(f"  t-statistic: {t_statistic:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significance level: alpha = 0.05")
    
    if p_value < 0.05:
        print("\nStatistically Significant (p < 0.05)")
        print("  We reject the null hypothesis.")
        print("  There IS a significant difference in sentiment between")
        print("  crisis and non-crisis speeches.")
    else:
        print("\nNot Statistically Significant (p >= 0.05)")
        print("  We fail to reject the null hypothesis.")
        print("  There is NO significant difference in sentiment between")
        print("  crisis and non-crisis speeches.")
    
    print()
    return t_statistic, p_value


# Compare sentiment classifications
def sentiment_classification_comparison(crisis, non_crisis):
    print("Sentiment Classification Comparison\n")
    
    print("Crisis Speeches:")
    crisis_sentiment_counts = crisis['overall_sentiment'].value_counts()
    for sentiment, count in crisis_sentiment_counts.items():
        pct = count / len(crisis) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    print("\nNon-Crisis Speeches:")
    non_crisis_sentiment_counts = non_crisis['overall_sentiment'].value_counts()
    for sentiment, count in non_crisis_sentiment_counts.items():
        pct = count / len(non_crisis) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    print()


# Create individual visualization graphs
def create_visualizations(df, crisis, non_crisis, t_stat, p_value):
    print("Creating visualizations...")
    
    sns.set_style("whitegrid")
    output_dir = os.path.join('..', 'OUTPUTS')
    
    # Graph 1: Box plot comparing positive percentages
    plt.figure(figsize=(10, 6))
    data_for_box = [crisis['positive_percentage'], non_crisis['positive_percentage']]
    bp = plt.boxplot(data_for_box, tick_labels=['Crisis', 'Non-Crisis'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    plt.ylabel('Positive Sentiment %', fontsize=12)
    plt.title('Distribution of Positive Sentiment: Crisis vs Non-Crisis', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_positive_sentiment.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: boxplot_positive_sentiment.png")
    
    # Graph 2: Histogram of positive percentages
    plt.figure(figsize=(10, 6))
    plt.hist(crisis['positive_percentage'], bins=20, alpha=0.6, label='Crisis', color='red', edgecolor='black')
    plt.hist(non_crisis['positive_percentage'], bins=20, alpha=0.6, label='Non-Crisis', color='blue', edgecolor='black')
    plt.xlabel('Positive Sentiment %', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Histogram of Positive Sentiment', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'histogram_positive_sentiment.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: histogram_positive_sentiment.png")
    
    # Graph 3: Crisis speeches sentiment classification pie chart
    plt.figure(figsize=(8, 8))
    crisis_sentiment = crisis['overall_sentiment'].value_counts()
    colors = ['#90EE90' if s == 'POSITIVE' else '#FFB6C6' for s in crisis_sentiment.index]
    plt.pie(crisis_sentiment.values, labels=crisis_sentiment.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Crisis Speeches Sentiment Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pie_crisis_sentiment.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: pie_crisis_sentiment.png")
    
    # Graph 4: Non-crisis speeches sentiment classification pie chart
    plt.figure(figsize=(8, 8))
    non_crisis_sentiment = non_crisis['overall_sentiment'].value_counts()
    colors = ['#90EE90' if s == 'POSITIVE' else '#FFB6C6' for s in non_crisis_sentiment.index]
    plt.pie(non_crisis_sentiment.values, labels=non_crisis_sentiment.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('Non-Crisis Speeches Sentiment Classification', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pie_noncrisis_sentiment.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: pie_noncrisis_sentiment.png")
    
    # Graph 5: Positive/Negative ratio comparison box plot
    plt.figure(figsize=(10, 6))
    crisis_ratios = crisis['positive_negative_ratio'].dropna()
    non_crisis_ratios = non_crisis['positive_negative_ratio'].dropna()
    data_for_box2 = [crisis_ratios, non_crisis_ratios]
    bp2 = plt.boxplot(data_for_box2, tick_labels=['Crisis', 'Non-Crisis'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightcoral')
    bp2['boxes'][1].set_facecolor('lightblue')
    plt.ylabel('Positive/Negative Ratio', fontsize=12)
    plt.title('Positive/Negative Ratio Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'boxplot_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: boxplot_ratio.png")
    
    # Graph 6: Time series plot
    plt.figure(figsize=(16, 6))
    crisis_data = df[df['is_crisis'] == True]
    non_crisis_data = df[df['is_crisis'] == False]
    
    plt.scatter(crisis_data['year'], crisis_data['positive_percentage'], 
               c='red', alpha=0.6, s=100, label='Crisis', marker='o', edgecolors='black')
    plt.scatter(non_crisis_data['year'], non_crisis_data['positive_percentage'],
               c='blue', alpha=0.6, s=100, label='Non-Crisis', marker='s', edgecolors='black')
    
    # Add trend lines
    z_crisis = np.polyfit(crisis_data['year'], crisis_data['positive_percentage'], 1)
    p_crisis = np.poly1d(z_crisis)
    plt.plot(crisis_data['year'], p_crisis(crisis_data['year']), "r--", alpha=0.8, linewidth=2, label='Crisis Trend')
    
    z_non = np.polyfit(non_crisis_data['year'], non_crisis_data['positive_percentage'], 1)
    p_non = np.poly1d(z_non)
    plt.plot(non_crisis_data['year'], p_non(non_crisis_data['year']), "b--", alpha=0.8, linewidth=2, label='Non-Crisis Trend')
    
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel('Positive Sentiment %', fontsize=12, fontweight='bold')
    plt.title('Positive Sentiment Over Time: Crisis vs Non-Crisis Speeches', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'timeseries_sentiment.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: timeseries_sentiment.png")
    
    print("Visualizations complete!\n")


# Save text summary report
def save_summary_report(df, crisis, non_crisis, t_stat, p_value):
    output_path = os.path.join('..', 'OUTPUTS', 'analysis_summary.txt')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("Crisis vs Non-Crisis SOTU Sentiment Analysis\n\n")
        
        f.write(f"Total speeches analyzed: {len(df)}\n")
        f.write(f"Date range: {df['year'].min()} - {df['year'].max()}\n\n")
        
        f.write("Descriptive Statistics:\n")
        f.write(f"Crisis speeches: {len(crisis)}\n")
        f.write(f"  Mean positive %: {crisis['positive_percentage'].mean():.2f}%\n")
        f.write(f"  Std dev: {crisis['positive_percentage'].std():.2f}%\n\n")
        
        f.write(f"Non-crisis speeches: {len(non_crisis)}\n")
        f.write(f"  Mean positive %: {non_crisis['positive_percentage'].mean():.2f}%\n")
        f.write(f"  Std dev: {non_crisis['positive_percentage'].std():.2f}%\n\n")
        
        f.write("Two-Sample T-Test:\n")
        f.write(f"t-statistic: {t_stat:.4f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
        f.write(f"Significance level: alpha = 0.05\n\n")
        
        if p_value < 0.05:
            f.write("Result: Statistically Significant\n")
            f.write("There IS a significant difference in sentiment between\n")
            f.write("crisis and non-crisis speeches.\n")
        else:
            f.write("Result: Not Statistically Significant\n")
            f.write("There is NO significant difference in sentiment between\n")
            f.write("crisis and non-crisis speeches.\n")
    
    print(f"Saved: analysis_summary.txt\n")


# Main analysis function
def main():
    print("\nCrisis vs Non-Crisis Sentiment Analysis\n")
    
    # Load and prepare data
    df = load_data()
    df, crisis, non_crisis = prepare_data(df)
    
    # Descriptive statistics
    descriptive_statistics(crisis, non_crisis)
    
    # Sentiment classification comparison
    sentiment_classification_comparison(crisis, non_crisis)
    
    # Perform t-test
    t_stat, p_value = perform_ttest(crisis, non_crisis)
    
    # Create visualizations
    create_visualizations(df, crisis, non_crisis, t_stat, p_value)
    
    # Save summary report
    save_summary_report(df, crisis, non_crisis, t_stat, p_value)
    
    print("Analysis Complete!")


if __name__ == "__main__":
    main()