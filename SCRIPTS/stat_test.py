#!/usr/bin/env python3
"""
Statistical Analysis: Crisis vs Non-Crisis State of the Union Speeches
Performs two-sample t-test and creates visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def load_data():
    """Load the sentiment analysis results"""
    print("Loading sentiment analysis results...")
    df = pd.read_csv('sentiment_analysis_results.csv')
    print(f"Loaded {len(df)} speeches from {df['year'].min()} to {df['year'].max()}\n")
    return df

def prepare_data(df):
    """Prepare data for analysis"""
    # Convert is_crisis to boolean
    df['is_crisis'] = df['is_crisis'].astype(bool)
    
    # Calculate ratio of positive to negative
    df['positive_negative_ratio'] = df['positive_count'] / df['negative_count']
    
    # Separate crisis and non-crisis speeches
    crisis = df[df['is_crisis'] == True].copy()
    non_crisis = df[df['is_crisis'] == False].copy()
    
    print(f"Crisis speeches: {len(crisis)}")
    print(f"Non-crisis speeches: {len(non_crisis)}\n")
    
    return df, crisis, non_crisis

def descriptive_statistics(crisis, non_crisis):
    """Calculate and display descriptive statistics"""
    print("="*60)
    print("DESCRIPTIVE STATISTICS")
    print("="*60)
    
    print("\nCRISIS SPEECHES:")
    print(f"  Count: {len(crisis)}")
    print(f"  Mean positive %: {crisis['positive_percentage'].mean():.2f}%")
    print(f"  Std dev: {crisis['positive_percentage'].std():.2f}%")
    print(f"  Median positive %: {crisis['positive_percentage'].median():.2f}%")
    print(f"  Min: {crisis['positive_percentage'].min():.2f}%")
    print(f"  Max: {crisis['positive_percentage'].max():.2f}%")
    
    print("\nNON-CRISIS SPEECHES:")
    print(f"  Count: {len(non_crisis)}")
    print(f"  Mean positive %: {non_crisis['positive_percentage'].mean():.2f}%")
    print(f"  Std dev: {non_crisis['positive_percentage'].std():.2f}%")
    print(f"  Median positive %: {non_crisis['positive_percentage'].median():.2f}%")
    print(f"  Min: {non_crisis['positive_percentage'].min():.2f}%")
    print(f"  Max: {non_crisis['positive_percentage'].max():.2f}%")
    
    print("\nDIFFERENCE:")
    diff = crisis['positive_percentage'].mean() - non_crisis['positive_percentage'].mean()
    print(f"  Crisis - Non-crisis: {diff:.2f} percentage points")
    print("="*60 + "\n")

def perform_ttest(crisis, non_crisis):
    """Perform two-sample t-test using positive/negative ratio"""
    print("="*60)
    print("TWO-SAMPLE T-TEST")
    print("="*60)
    
    # Get positive/negative ratios
    crisis_ratios = crisis['positive_negative_ratio'].dropna()
    non_crisis_ratios = non_crisis['positive_negative_ratio'].dropna()
    
    print(f"\nUsing positive/negative ratio as the metric")
    print(f"Crisis speeches: n={len(crisis_ratios)}, mean ratio={crisis_ratios.mean():.3f}")
    print(f"Non-crisis speeches: n={len(non_crisis_ratios)}, mean ratio={non_crisis_ratios.mean():.3f}")
    
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(crisis_ratios, non_crisis_ratios)
    
    print(f"\nResults:")
    print(f"  t-statistic: {t_statistic:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Significance level: α = 0.05")
    
    if p_value < 0.05:
        print(f"\n✓ STATISTICALLY SIGNIFICANT (p < 0.05)")
        print(f"  We reject the null hypothesis.")
        print(f"  There IS a significant difference in sentiment between")
        print(f"  crisis and non-crisis speeches.")
    else:
        print(f"\n✗ NOT STATISTICALLY SIGNIFICANT (p >= 0.05)")
        print(f"  We fail to reject the null hypothesis.")
        print(f"  There is NO significant difference in sentiment between")
        print(f"  crisis and non-crisis speeches.")
    
    print("="*60 + "\n")
    
    return t_statistic, p_value

def sentiment_classification_comparison(crisis, non_crisis):
    """Compare sentiment classifications between crisis and non-crisis"""
    print("="*60)
    print("SENTIMENT CLASSIFICATION COMPARISON")
    print("="*60)
    
    print("\nCRISIS SPEECHES:")
    crisis_sentiment_counts = crisis['overall_sentiment'].value_counts()
    for sentiment, count in crisis_sentiment_counts.items():
        pct = count / len(crisis) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    print("\nNON-CRISIS SPEECHES:")
    non_crisis_sentiment_counts = non_crisis['overall_sentiment'].value_counts()
    for sentiment, count in non_crisis_sentiment_counts.items():
        pct = count / len(non_crisis) * 100
        print(f"  {sentiment}: {count} ({pct:.1f}%)")
    
    print("="*60 + "\n")

def create_visualizations(df, crisis, non_crisis, t_stat, p_value):
    """Create comprehensive visualizations"""
    print("Creating visualizations...")
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Crisis vs Non-Crisis SOTU Speech Sentiment Analysis', fontsize=16, fontweight='bold')
    
    # 1. Box plot comparing positive percentages
    ax1 = axes[0, 0]
    data_for_box = [crisis['positive_percentage'], non_crisis['positive_percentage']]
    bp = ax1.boxplot(data_for_box, labels=['Crisis', 'Non-Crisis'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax1.set_ylabel('Positive Sentiment %', fontsize=12)
    ax1.set_title('Distribution of Positive Sentiment', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Histogram of positive percentages
    ax2 = axes[0, 1]
    ax2.hist(crisis['positive_percentage'], bins=20, alpha=0.6, label='Crisis', color='red', edgecolor='black')
    ax2.hist(non_crisis['positive_percentage'], bins=20, alpha=0.6, label='Non-Crisis', color='blue', edgecolor='black')
    ax2.set_xlabel('Positive Sentiment %', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Histogram of Positive Sentiment', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Sentiment classification pie charts
    ax3 = axes[0, 2]
    crisis_sentiment = crisis['overall_sentiment'].value_counts()
    colors = ['#90EE90' if s == 'POSITIVE' else '#FFB6C6' for s in crisis_sentiment.index]
    ax3.pie(crisis_sentiment.values, labels=crisis_sentiment.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    ax3.set_title('Crisis Speeches\nSentiment Classification', fontsize=12, fontweight='bold')
    
    # 4. Non-crisis sentiment classification
    ax4 = axes[1, 0]
    non_crisis_sentiment = non_crisis['overall_sentiment'].value_counts()
    colors = ['#90EE90' if s == 'POSITIVE' else '#FFB6C6' for s in non_crisis_sentiment.index]
    ax4.pie(non_crisis_sentiment.values, labels=non_crisis_sentiment.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax4.set_title('Non-Crisis Speeches\nSentiment Classification', fontsize=12, fontweight='bold')
    
    # 5. Positive/Negative Ratio comparison
    ax5 = axes[1, 1]
    crisis_ratios = crisis['positive_negative_ratio'].dropna()
    non_crisis_ratios = non_crisis['positive_negative_ratio'].dropna()
    data_for_box2 = [crisis_ratios, non_crisis_ratios]
    bp2 = ax5.boxplot(data_for_box2, labels=['Crisis', 'Non-Crisis'], patch_artist=True)
    bp2['boxes'][0].set_facecolor('lightcoral')
    bp2['boxes'][1].set_facecolor('lightblue')
    ax5.set_ylabel('Positive/Negative Ratio', fontsize=12)
    ax5.set_title('Positive/Negative Ratio Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. T-test results summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Create text summary
    summary_text = f"""
    TWO-SAMPLE T-TEST RESULTS
    
    Metric: Positive/Negative Ratio
    
    Crisis Speeches:
      n = {len(crisis_ratios)}
      Mean = {crisis_ratios.mean():.3f}
      SD = {crisis_ratios.std():.3f}
    
    Non-Crisis Speeches:
      n = {len(non_crisis_ratios)}
      Mean = {non_crisis_ratios.mean():.3f}
      SD = {non_crisis_ratios.std():.3f}
    
    t-statistic = {t_stat:.4f}
    p-value = {p_value:.4f}
    α = 0.05
    
    """
    
    if p_value < 0.05:
        summary_text += "    Result: SIGNIFICANT ✓\n    (p < 0.05)"
        result_color = 'green'
    else:
        summary_text += "    Result: NOT SIGNIFICANT\n    (p ≥ 0.05)"
        result_color = 'red'
    
    ax6.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig('crisis_vs_noncrisis_analysis.png', dpi=300, bbox_inches='tight')
    print("  Saved: crisis_vs_noncrisis_analysis.png")
    
    # Create time series plot
    fig2, ax = plt.subplots(figsize=(16, 6))
    
    # Plot positive percentage over time
    crisis_data = df[df['is_crisis'] == True]
    non_crisis_data = df[df['is_crisis'] == False]
    
    ax.scatter(crisis_data['year'], crisis_data['positive_percentage'], 
               c='red', alpha=0.6, s=100, label='Crisis', marker='o', edgecolors='black')
    ax.scatter(non_crisis_data['year'], non_crisis_data['positive_percentage'],
               c='blue', alpha=0.6, s=100, label='Non-Crisis', marker='s', edgecolors='black')
    
    # Add trend lines
    z_crisis = np.polyfit(crisis_data['year'], crisis_data['positive_percentage'], 1)
    p_crisis = np.poly1d(z_crisis)
    ax.plot(crisis_data['year'], p_crisis(crisis_data['year']), "r--", alpha=0.8, linewidth=2, label='Crisis Trend')
    
    z_non = np.polyfit(non_crisis_data['year'], non_crisis_data['positive_percentage'], 1)
    p_non = np.poly1d(z_non)
    ax.plot(non_crisis_data['year'], p_non(non_crisis_data['year']), "b--", alpha=0.8, linewidth=2, label='Non-Crisis Trend')
    
    ax.set_xlabel('Year', fontsize=12, fontweight='bold')
    ax.set_ylabel('Positive Sentiment %', fontsize=12, fontweight='bold')
    ax.set_title('Positive Sentiment Over Time: Crisis vs Non-Crisis Speeches', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sentiment_over_time.png', dpi=300, bbox_inches='tight')
    print("  Saved: sentiment_over_time.png")
    
    plt.close('all')
    print("Visualizations complete!\n")

def save_summary_report(df, crisis, non_crisis, t_stat, p_value):
    """Save a text summary report"""
    with open('analysis_summary.txt', 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("CRISIS VS NON-CRISIS SOTU SENTIMENT ANALYSIS\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total speeches analyzed: {len(df)}\n")
        f.write(f"Date range: {df['year'].min()} - {df['year'].max()}\n\n")
        
        f.write("DESCRIPTIVE STATISTICS:\n")
        f.write("-"*60 + "\n")
        f.write(f"Crisis speeches: {len(crisis)}\n")
        f.write(f"  Mean positive %: {crisis['positive_percentage'].mean():.2f}%\n")
        f.write(f"  Std dev: {crisis['positive_percentage'].std():.2f}%\n\n")
        
        f.write(f"Non-crisis speeches: {len(non_crisis)}\n")
        f.write(f"  Mean positive %: {non_crisis['positive_percentage'].mean():.2f}%\n")
        f.write(f"  Std dev: {non_crisis['positive_percentage'].std():.2f}%\n\n")
        
        f.write("TWO-SAMPLE T-TEST:\n")
        f.write("-"*60 + "\n")
        f.write(f"t-statistic: {t_stat:.4f}\n")
        f.write(f"p-value: {p_value:.4f}\n")
        f.write(f"Significance level: alpha = 0.05\n\n")
        
        if p_value < 0.05:
            f.write("RESULT: STATISTICALLY SIGNIFICANT\n")
            f.write("There IS a significant difference in sentiment between\n")
            f.write("crisis and non-crisis speeches.\n")
        else:
            f.write("RESULT: NOT STATISTICALLY SIGNIFICANT\n")
            f.write("There is NO significant difference in sentiment between\n")
            f.write("crisis and non-crisis speeches.\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print("Saved: analysis_summary.txt\n")

def main():
    print("="*60)
    print("CRISIS VS NON-CRISIS SENTIMENT ANALYSIS")
    print("="*60 + "\n")
    
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
    
    print("="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  1. crisis_vs_noncrisis_analysis.png")
    print("  2. sentiment_over_time.png")
    print("  3. analysis_summary.txt")

if __name__ == "__main__":
    main()