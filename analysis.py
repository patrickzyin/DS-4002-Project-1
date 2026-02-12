#!/usr/bin/env python3
"""
Analyze all State of the Union speeches from 1850 to 2025
Groups by title carefully (only combining actual continuations like "Part 2")
Uses percentile-based classification: bottom 25% = negative, top 25% = positive
Saves results to CSV for later analysis
"""

from transformers import pipeline
import pandas as pd
import re
import ast
import numpy as np

def parse_speech_text(text):
    """
    Parse the speech text which might be stored as a string representation of a list.
    
    Args:
        text: Raw text from CSV (might be a list string like "['sentence 1', 'sentence 2']")
    
    Returns:
        Cleaned text as a single string
    """
    try:
        # Try to parse as a Python list
        speech_list = ast.literal_eval(text)
        if isinstance(speech_list, list):
            # Join all sentences with spaces
            return ' '.join(speech_list)
    except:
        # If it fails, just return the text as is
        pass
    
    return text

def clean_speech_text(text):
    """
    Remove annotations like (applause), [laughter], etc. from speech text.
    
    Args:
        text: Raw speech text
    
    Returns:
        Cleaned text without annotations
    """
    # Remove text in parentheses: (applause), (laughter), etc.
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove text in square brackets: [applause], [laughter], etc.
    text = re.sub(r'\[(?!\'[^\]]*$)[^\]]*\]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def normalize_title(title):
    """
    Normalize title by removing 'Part 2:', 'Part 3:', etc. to identify base speech.
    
    Args:
        title: Original title
    
    Returns:
        Normalized title without part indicators
    """
    # Remove "Part X:" or "Part X -" from the beginning
    normalized = re.sub(r'^Part\s+\d+[\s:‐\-–—]+', '', title, flags=re.IGNORECASE)
    return normalized.strip()

def is_continuation(title):
    """
    Check if a title indicates a continuation (Part 2, Part 3, etc.)
    
    Args:
        title: Title to check
    
    Returns:
        True if it's a continuation, False otherwise
    """
    return bool(re.match(r'^Part\s+\d+', title, flags=re.IGNORECASE))

def analyze_speech_sentiment(speech_text):
    """
    Analyze sentiment of a State of the Union speech.
    
    Args:
        speech_text: The full text of the speech
    
    Returns:
        Dictionary with overall sentiment and detailed results
    """
    # First parse the text format
    parsed_text = parse_speech_text(speech_text)
    
    # Clean the text
    cleaned_text = clean_speech_text(parsed_text)
    
    # Split speech into sentences for more granular analysis
    sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
    
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
                print(f"      Error analyzing sentence {i}: {e}")
    
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
        'positive_scores': positive_scores,
        'negative_scores': negative_scores
    }

def combine_sentiment_results(results_list):
    """
    Combine sentiment results from multiple speech parts.
    
    Args:
        results_list: List of sentiment analysis results dictionaries
    
    Returns:
        Combined sentiment results
    """
    # Aggregate counts
    total_sentences = sum(r['total_sentences'] for r in results_list)
    positive_count = sum(r['positive_count'] for r in results_list)
    negative_count = sum(r['negative_count'] for r in results_list)
    
    # Combine all scores
    all_positive_scores = []
    all_negative_scores = []
    
    for r in results_list:
        all_positive_scores.extend(r['positive_scores'])
        all_negative_scores.extend(r['negative_scores'])
    
    # Calculate combined statistics
    positive_percentage = (positive_count / total_sentences * 100) if total_sentences > 0 else 0
    negative_percentage = (negative_count / total_sentences * 100) if total_sentences > 0 else 0
    
    avg_positive_confidence = sum(all_positive_scores) / len(all_positive_scores) if all_positive_scores else 0
    avg_negative_confidence = sum(all_negative_scores) / len(all_negative_scores) if all_negative_scores else 0
    
    return {
        'total_sentences': total_sentences,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'positive_percentage': positive_percentage,
        'negative_percentage': negative_percentage,
        'avg_positive_confidence': avg_positive_confidence,
        'avg_negative_confidence': avg_negative_confidence
    }

def classify_by_percentile(positive_percentages):
    """
    Classify speeches based on percentiles of positive sentiment ratios.
    
    Uses the distribution of positive percentages across all speeches:
    - Bottom 25% (≤ 25th percentile) → NEGATIVE
    - Top 25% (≥ 75th percentile) → POSITIVE  
    - Middle 50% (between 25th-75th) → POSITIVE if ≥ median, NEGATIVE if < median
    
    No neutral category - everything is classified as either POSITIVE or NEGATIVE.
    
    Args:
        positive_percentages: List of positive percentages for all speeches
    
    Returns:
        Tuple of (classifications list, 25th percentile threshold, 75th percentile threshold)
    """
    # Calculate percentile thresholds from the distribution
    p25 = np.percentile(positive_percentages, 25)
    p50 = np.percentile(positive_percentages, 50)  # median
    p75 = np.percentile(positive_percentages, 75)
    
    print(f"\nPercentile thresholds calculated from distribution:")
    print(f"  25th percentile (bottom 25%): {p25:.2f}% positive")
    print(f"  50th percentile (median): {p50:.2f}% positive")
    print(f"  75th percentile (top 25%): {p75:.2f}% positive")
    print(f"\nClassification rules (no neutrals):")
    print(f"  NEGATIVE: positive_percentage ≤ {p25:.2f}% OR ({p25:.2f}% < positive_percentage < {p50:.2f}%)")
    print(f"  POSITIVE: positive_percentage ≥ {p75:.2f}% OR ({p50:.2f}% ≤ positive_percentage < {p75:.2f}%)")
    print(f"  Essentially: < {p50:.2f}% = NEGATIVE, ≥ {p50:.2f}% = POSITIVE\n")
    
    classifications = []
    for pct in positive_percentages:
        # Use median as the dividing line
        if pct >= p50:
            classifications.append('POSITIVE')
        else:
            classifications.append('NEGATIVE')
    
    return classifications, p25, p75

def main():
    global sentiment_analyzer
    
    print("="*60)
    print("STATE OF THE UNION SENTIMENT ANALYSIS")
    print("Analyzing speeches from 1850 to 2025")
    print("Carefully grouping only true continuations (Part 2, etc.)")
    print("Using percentile-based classification")
    print("="*60)
    
    print("\nReading SOTU_data_final.csv...")
    
    df = pd.read_csv('SOTU_data_final.csv', 
                     quotechar='"',
                     encoding='utf-8',
                     on_bad_lines='skip')
    
    print(f"Successfully read CSV with {len(df)} total speeches\n")
    
    # Filter for years 1850-2025
    df_filtered = df[(df['Year'] >= 1850) & (df['Year'] <= 2025)].copy()
    print(f"Found {len(df_filtered)} speech rows between 1850-2025")
    
    # Create a unique identifier for each speech
    # Group by year + normalized title, but only if it's a continuation
    df_filtered['normalized_title'] = df_filtered['Title'].apply(normalize_title)
    df_filtered['is_continuation'] = df_filtered['Title'].apply(is_continuation)
    
    # Create grouping key: Year_NormalizedTitle if continuation, otherwise Year_OriginalTitle
    def create_group_key(row):
        if row['is_continuation']:
            # This is a continuation - group with base speech
            return f"{int(row['Year'])}_{row['normalized_title']}"
        else:
            # This is a unique speech - use original title to keep it separate
            return f"{int(row['Year'])}_{row['Title']}"
    
    df_filtered['group_key'] = df_filtered.apply(create_group_key, axis=1)
    
    # Get unique speeches
    unique_groups = df_filtered['group_key'].unique()
    print(f"Identified {len(unique_groups)} unique speeches (after combining continuations)\n")
    
    # Load the model once
    print("Loading sentiment analysis model...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Model loaded!\n")
    
    print("="*60)
    print("STARTING ANALYSIS")
    print("="*60)
    
    # Store results
    all_results = []
    
    for speech_num, group_key in enumerate(sorted(unique_groups), 1):
        # Get all parts for this speech
        speech_parts = df_filtered[df_filtered['group_key'] == group_key].sort_values('Title')
        
        # Get info from first row
        first_speech = speech_parts.iloc[0]
        year = first_speech['Year']
        president = first_speech['President']
        
        # Use normalized title if it's a continuation, otherwise original
        if first_speech['is_continuation']:
            display_title = first_speech['normalized_title']
        else:
            display_title = first_speech['Title']
        
        # Check for additional columns if they exist
        is_crisis = first_speech['is_crisis'] if 'is_crisis' in speech_parts.columns else None
        crisis_name = first_speech['crisis_name'] if 'crisis_name' in speech_parts.columns else None
        
        print(f"\n[{speech_num}/{len(unique_groups)}] Analyzing {int(year)} - {president}")
        print(f"  Title: {display_title[:80]}...")
        
        if len(speech_parts) > 1:
            print(f"  Found {len(speech_parts)} parts - will combine:")
            for idx, part in speech_parts.iterrows():
                print(f"    - {part['Title'][:60]}")
        
        # Analyze each part
        part_results = []
        total_chars = 0
        
        for part_num, (idx, speech) in enumerate(speech_parts.iterrows(), 1):
            text = str(speech['Text'])
            
            # Check if text is valid
            if text == 'nan' or text == '' or len(text) < 100:
                print(f"  Warning: Part {part_num} has no valid text")
                continue
            
            total_chars += len(text)
            
            if len(speech_parts) > 1:
                print(f"    Analyzing part {part_num}/{len(speech_parts)}... ({len(text)} chars)")
            
            # Perform sentiment analysis
            analysis = analyze_speech_sentiment(text)
            part_results.append(analysis)
        
        if not part_results:
            print(f"  Error: No valid text found")
            continue
        
        # Combine results if multiple parts
        if len(part_results) > 1:
            combined_analysis = combine_sentiment_results(part_results)
        else:
            combined_analysis = part_results[0]
        
        # Store result (classification will be done after all speeches are analyzed)
        result = {
            'year': int(year),
            'president': president,
            'title': display_title,
            'total_sentences': combined_analysis['total_sentences'],
            'positive_count': combined_analysis['positive_count'],
            'negative_count': combined_analysis['negative_count'],
            'positive_percentage': round(combined_analysis['positive_percentage'], 2),
            'negative_percentage': round(combined_analysis['negative_percentage'], 2),
            'avg_positive_confidence': round(combined_analysis['avg_positive_confidence'], 4),
            'avg_negative_confidence': round(combined_analysis['avg_negative_confidence'], 4),
            'num_parts': len(speech_parts),
            'total_characters': total_chars
        }
        
        # Add crisis info if available
        if is_crisis is not None:
            result['is_crisis'] = is_crisis
        if crisis_name is not None:
            result['crisis_name'] = crisis_name
        
        all_results.append(result)
        
        print(f"  ✓ Complete: {combined_analysis['total_sentences']} sentences, "
              f"{combined_analysis['positive_percentage']:.1f}% positive")
    
    # Create DataFrame with results
    results_df = pd.DataFrame(all_results)
    
    # Sort by year
    results_df = results_df.sort_values('year').reset_index(drop=True)
    
    # Now classify based on percentiles
    print("\n" + "="*60)
    print("CALCULATING PERCENTILE-BASED CLASSIFICATIONS")
    print("="*60)
    
    classifications, p25, p75 = classify_by_percentile(results_df['positive_percentage'].tolist())
    results_df['overall_sentiment'] = classifications
    results_df['p25_threshold'] = p25
    results_df['p75_threshold'] = p75
    
    # Save to CSV
    output_filename = 'sentiment_analysis_results.csv'
    results_df.to_csv(output_filename, index=False)
    
    print("="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Analyzed {len(results_df)} speeches from {results_df['year'].min()} to {results_df['year'].max()}")
    print(f"Results saved to: {output_filename}")
    print("\nSummary statistics:")
    print(f"  Average positive sentiment: {results_df['positive_percentage'].mean():.2f}%")
    print(f"  Average negative sentiment: {results_df['negative_percentage'].mean():.2f}%")
    print(f"  Std dev of positive sentiment: {results_df['positive_percentage'].std():.2f}%")
    
    # Count sentiment classifications
    sentiment_counts = results_df['overall_sentiment'].value_counts()
    print(f"\nSentiment distribution (percentile-based, no neutrals):")
    for sentiment in ['POSITIVE', 'NEGATIVE']:
        if sentiment in sentiment_counts:
            count = sentiment_counts[sentiment]
            print(f"  {sentiment}: {count} speeches ({count/len(results_df)*100:.1f}%)")
    
    print(f"\nMost positive speech: {results_df.loc[results_df['positive_percentage'].idxmax(), 'year']} "
          f"({results_df['positive_percentage'].max():.1f}%) - {results_df.loc[results_df['positive_percentage'].idxmax(), 'president']}")
    print(f"Most negative speech: {results_df.loc[results_df['positive_percentage'].idxmin(), 'year']} "
          f"({results_df['positive_percentage'].min():.1f}%) - {results_df.loc[results_df['positive_percentage'].idxmin(), 'president']}")
    print("="*60)
    
    # Show preview
    print("\nPreview of results:")
    print(results_df[['year', 'president', 'positive_percentage', 'overall_sentiment']].head(10))

if __name__ == "__main__":
    main()