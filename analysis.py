#!/usr/bin/env python3
"""
Analyze the 2003 and 2004 State of the Union speeches
"""

from transformers import pipeline
import pandas as pd
import re

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
    text = re.sub(r'\[[^\]]*\]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def analyze_speech_sentiment(speech_text):
    """
    Analyze sentiment of a State of the Union speech.
    
    Args:
        speech_text: The full text of the speech
    
    Returns:
        Dictionary with overall sentiment and detailed results
    """
    # Clean the text first
    print("  Cleaning text (removing annotations like (applause), [laughter], etc.)...")
    cleaned_text = clean_speech_text(speech_text)
    print(f"  Original text length: {len(speech_text)} characters")
    print(f"  Cleaned text length: {len(cleaned_text)} characters")
    
    # Split speech into sentences for more granular analysis
    sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
    
    print(f"  Analyzing {len(sentences)} sentences...")
    
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
                
                # Show progress every 50 sentences
                if (i + 1) % 50 == 0:
                    print(f"    Processed {i + 1} sentences...")
                    
            except Exception as e:
                print(f"  Error analyzing sentence {i}: {e}")
    
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
        'detailed_results': results
    }

def print_speech_results(speech_info, analysis):
    """Print results for a single speech"""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS RESULTS")
    print("="*60)
    print(f"President: {speech_info['President']}")
    print(f"Year: {speech_info['Year']}")
    print(f"Title: {speech_info['Title']}")
    print("-"*60)
    print(f"Total sentences analyzed: {analysis['total_sentences']}")
    print(f"\nPositive sentences: {analysis['positive_count']} ({analysis['positive_percentage']:.1f}%)")
    print(f"Negative sentences: {analysis['negative_count']} ({analysis['negative_percentage']:.1f}%)")
    print(f"\nAverage positive confidence: {analysis['avg_positive_confidence']:.3f}")
    print(f"Average negative confidence: {analysis['avg_negative_confidence']:.3f}")
    
    # Overall sentiment
    if analysis['positive_percentage'] > analysis['negative_percentage']:
        overall = "POSITIVE"
    elif analysis['negative_percentage'] > analysis['positive_percentage']:
        overall = "NEGATIVE"
    else:
        overall = "NEUTRAL"
    
    print(f"\nOverall speech sentiment: {overall}")
    print("="*60)
    
    # Show most positive and negative sentences
    sorted_results = sorted(
        analysis['detailed_results'],
        key=lambda x: x['score'],
        reverse=True
    )
    
    print("\nTOP 3 MOST POSITIVE SENTENCES")
    print("-"*60)
    positive_sentences = [r for r in sorted_results if r['label'] == 'POSITIVE'][:3]
    for i, result in enumerate(positive_sentences, 1):
        print(f"\n{i}. {result['sentence'][:150]}...")
        print(f"   Confidence: {result['score']:.3f}")
    
    print("\n" + "-"*60)
    print("TOP 3 MOST NEGATIVE SENTENCES")
    print("-"*60)
    negative_sentences = [r for r in sorted_results if r['label'] == 'NEGATIVE'][:3]
    for i, result in enumerate(negative_sentences, 1):
        print(f"\n{i}. {result['sentence'][:150]}...")
        print(f"   Confidence: {result['score']:.3f}")
    print()

def main():
    global sentiment_analyzer
    
    print("Reading SOTU_data_final.csv...")
    
    df = pd.read_csv('SOTU_data_final.csv', 
                     quotechar='"',
                     encoding='utf-8',
                     on_bad_lines='skip')
    
    print(f"Successfully read CSV with {len(df)} speeches\n")
    
    # Load the model once for both analyses
    print("Loading sentiment analysis model...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Model loaded!\n")
    
    # Years to analyze
    years_to_analyze = [2003, 2004]
    
    results_summary = []
    
    for year in years_to_analyze:
        # Filter for the year
        speech_df = df[df['Year'] == year]
        
        if len(speech_df) == 0:
            print(f"Warning: No speech found for year {year}!")
            continue
        
        # Get the speech
        speech = speech_df.iloc[0]
        
        print("="*60)
        print(f"ANALYZING {year} STATE OF THE UNION SPEECH")
        print("="*60)
        print(f"President: {speech['President']}")
        print(f"Year: {speech['Year']}")
        print(f"Title: {speech['Title']}")
        print("="*60)
        
        # Get the text
        text = str(speech['Text'])
        
        # Check if text is valid
        if text == 'nan' or text == '' or len(text) < 100:
            print(f"Error: Text appears to be empty or invalid (length: {len(text)})\n")
            continue
        
        # Perform sentiment analysis
        analysis = analyze_speech_sentiment(text)
        
        # Store summary
        results_summary.append({
            'year': year,
            'president': speech['President'],
            'positive_pct': analysis['positive_percentage'],
            'negative_pct': analysis['negative_percentage'],
            'total_sentences': analysis['total_sentences']
        })
        
        # Print detailed results
        print_speech_results(speech, analysis)
    
    # Print comparison
    if len(results_summary) == 2:
        print("\n" + "="*60)
        print("COMPARISON: 2003 vs 2004")
        print("="*60)
        for result in results_summary:
            print(f"\n{result['year']} ({result['president']}):")
            print(f"  Positive: {result['positive_pct']:.1f}%")
            print(f"  Negative: {result['negative_pct']:.1f}%")
            print(f"  Total sentences: {result['total_sentences']}")
        
        # Show difference
        diff = results_summary[1]['positive_pct'] - results_summary[0]['positive_pct']
        print(f"\nChange from 2003 to 2004:")
        if diff > 0:
            print(f"  Positive sentiment INCREASED by {diff:.1f} percentage points")
        elif diff < 0:
            print(f"  Positive sentiment DECREASED by {abs(diff):.1f} percentage points")
        else:
            print(f"  No change in positive sentiment")
        print("="*60)

if __name__ == "__main__":
    main()