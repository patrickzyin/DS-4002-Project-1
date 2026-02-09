#!/usr/bin/env python3
"""
State of the Union Speech Sentiment Analysis
Reads speeches from a CSV file and performs sentiment analysis
"""

from transformers import pipeline
import pandas as pd
import sys
import os

def analyze_speech_sentiment(speech_text, chunk_size=512):
   
    # Using a robust sentiment analysis model
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    # Split speech into sentences for more granular analysis
    sentences = [s.strip() for s in speech_text.split('.') if s.strip()]
    
    results = []
    positive_count = 0
    negative_count = 0
    positive_scores = []
    negative_scores = []
    
    for i, sentence in enumerate(sentences):
        if len(sentence) > 10:  # Skip very short fragments
            try:
                result = sentiment_analyzer(sentence[:512])[0]  # Truncate if needed
                results.append({
                    'sentence': sentence,
                    'label': result['label'],
                    'score': result['score']
                })
                
                if result['label'] == 'POSITIVE':
                    positive_count += 1
                    positive_scores.append(result['score'])
                else:
                    negative_count += 1
                    negative_scores.append(result['score'])
                    
            except Exception as e:
                print(f"Error analyzing sentence {i}: {e}")
    
    # Calculate overall statistics
    total_analyzed = positive_count + negative_count
    positive_percentage = (positive_count / total_analyzed * 100) if total_analyzed > 0 else 0
    negative_percentage = (negative_count / total_analyzed * 100) if total_analyzed > 0 else 0
    
    avg_positive_confidence = sum(positive_scores) / len(positive_scores) if positive_scores else 0
    avg_negative_confidence = sum(negative_scores) / len(negative_scores) if negative_scores else 0
    
    return {
        'total_sentences': total_analyzed,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'positive_percentage': positive_percentage,
        'negative_percentage': negative_percentage,
        'avg_positive_confidence': avg_positive_confidence,
        'avg_negative_confidence': avg_negative_confidence,
        'overall_sentiment': 'POSITIVE' if positive_percentage > negative_percentage else 'NEGATIVE',
        'detailed_results': results
    }

def analyze_csv(csv_file, text_column='speech', output_file=None):
    """
    Analyze sentiments for all speeches in a CSV file.
    
    Args:
        csv_file: Path to CSV file
        text_column: Name of the column containing speech text
        output_file: Optional path to save results CSV
    
    Returns:
        DataFrame with sentiment analysis results
    """
    print(f"Reading CSV file: {csv_file}")
    df = pd.read_csv(csv_file)
    
    if text_column not in df.columns:
        print(f"\nError: Column '{text_column}' not found in CSV.")
        print(f"Available columns: {', '.join(df.columns)}")
        return None
    
    print(f"Found {len(df)} speeches to analyze")
    print("Loading sentiment analysis model...")
    
    # Store results
    results = []
    
    for idx, row in df.iterrows():
        speech_text = str(row[text_column])
        
        print(f"\n{'='*60}")
        print(f"Analyzing speech {idx + 1}/{len(df)}")
        
        # Show other columns if they exist (like year, president, etc.)
        for col in df.columns:
            if col != text_column:
                print(f"{col}: {row[col]}")
        print('='*60)
        
        # Analyze sentiment
        analysis = analyze_speech_sentiment(speech_text)
        
        # Create result dictionary with all original columns plus sentiment data
        result = row.to_dict()
        result['total_sentences'] = analysis['total_sentences']
        result['positive_count'] = analysis['positive_count']
        result['negative_count'] = analysis['negative_count']
        result['positive_percentage'] = round(analysis['positive_percentage'], 2)
        result['negative_percentage'] = round(analysis['negative_percentage'], 2)
        result['avg_positive_confidence'] = round(analysis['avg_positive_confidence'], 3)
        result['avg_negative_confidence'] = round(analysis['avg_negative_confidence'], 3)
        result['overall_sentiment'] = analysis['overall_sentiment']
        
        results.append(result)
        
        # Print summary
        print(f"\nResults:")
        print(f"  Total sentences: {analysis['total_sentences']}")
        print(f"  Positive: {analysis['positive_count']} ({analysis['positive_percentage']:.1f}%)")
        print(f"  Negative: {analysis['negative_count']} ({analysis['negative_percentage']:.1f}%)")
        print(f"  Overall sentiment: {analysis['overall_sentiment']}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Save to file if specified
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\n{'='*60}")
        print(f"Results saved to: {output_file}")
        print('='*60)
    
    return results_df

def print_summary_statistics(results_df):
    """Print overall summary statistics across all speeches"""
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY STATISTICS")
    print('='*60)
    print(f"Total speeches analyzed: {len(results_df)}")
    print(f"\nAverage positive percentage: {results_df['positive_percentage'].mean():.2f}%")
    print(f"Average negative percentage: {results_df['negative_percentage'].mean():.2f}%")
    print(f"\nMost positive speech: {results_df['positive_percentage'].max():.2f}%")
    print(f"Most negative speech: {results_df['negative_percentage'].max():.2f}%")
    
    overall_positive = (results_df['overall_sentiment'] == 'POSITIVE').sum()
    overall_negative = (results_df['overall_sentiment'] == 'NEGATIVE').sum()
    print(f"\nSpeeches with positive sentiment: {overall_positive}")
    print(f"Speeches with negative sentiment: {overall_negative}")
    print('='*60)

def main():
    if len(sys.argv) < 2:
        print("Usage: python sentiment_csv.py <csv_file> [text_column] [output_file]")
        print("\nExample:")
        print("  python sentiment_csv.py speeches.csv")
        print("  python sentiment_csv.py speeches.csv speech_text")
        print("  python sentiment_csv.py speeches.csv speech_text results.csv")
        print("\nThe CSV should have at least one column with the speech text.")
        print("By default, it looks for a column named 'speech'.")
        return
    
    csv_file = sys.argv[1]
    text_column = sys.argv[2] if len(sys.argv) > 2 else 'speech'
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        return
    
    # Analyze all speeches
    results_df = analyze_csv(csv_file, text_column, output_file)
    
    if results_df is not None:
        # Print summary statistics
        print_summary_statistics(results_df)
        
        # Show preview of results
        print("\nResults preview:")
        print(results_df[['positive_percentage', 'negative_percentage', 'overall_sentiment']].head())

if __name__ == "__main__":
    main()