#!/usr/bin/env python3
"""
TEST: Analyze ONLY 2010 and 2011 State of the Union speeches
"""

from transformers import pipeline
import pandas as pd
import re
import ast
import numpy as np
import csv as csv_module
import os

def parse_speech_text(text):
    """Parse the speech text which might be stored as a string representation of a list."""
    try:
        speech_list = ast.literal_eval(text)
        if isinstance(speech_list, list):
            return ' '.join(speech_list)
    except:
        pass
    return text

def clean_speech_text(text):
    """Remove annotations like (applause), [laughter], etc. from speech text."""
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[(?!\'[^\]]*$)[^\]]*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def analyze_speech_sentiment(speech_text):
    """Analyze sentiment of a State of the Union speech."""
    parsed_text = parse_speech_text(speech_text)
    cleaned_text = clean_speech_text(parsed_text)
    sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
    
    results = []
    positive_count = 0
    negative_count = 0
    positive_scores = []
    negative_scores = []
    
    for i, sentence in enumerate(sentences):
        if len(sentence) > 10:
            try:
                result = sentiment_analyzer(sentence[:512])[0]
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

def main():
    global sentiment_analyzer
    
    print("="*60)
    print("TEST: ANALYZING ONLY 2010 AND 2011")
    print("="*60)
    
    csv_path = 'SOTU_data_final.csv'
    print(f"\nReading from: {os.path.abspath(csv_path)}")
    
    # Read CSV with custom parser
    print("Using custom CSV parser...")
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv_module.reader(f, quotechar='"', delimiter=',')
        header = next(csv_reader)
        
        for row in csv_reader:
            if len(row) >= len(header):
                rows.append(row[:len(header)])
            elif len(row) >= 3:
                while len(row) < len(header):
                    row.append('')
                rows.append(row)
    
    df = pd.DataFrame(rows, columns=header)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    print(f"Successfully read CSV with {len(df)} total speeches")
    print(f"Columns: {list(df.columns)}\n")
    
    # Filter for ONLY 2010 and 2011
    df_filtered = df[(df['Year'] == 2010) | (df['Year'] == 2011)].copy()
    print(f"Found {len(df_filtered)} speeches for 2010-2011\n")
    
    if len(df_filtered) == 0:
        print("ERROR: No speeches found for 2010 or 2011!")
        print("\nShowing all Obama speeches in the data:")
        obama = df[df['President'].str.contains('Obama', na=False)]
        for idx, row in obama.iterrows():
            print(f"  Year: {row['Year']}, Title: {row['Title'][:60]}")
        return
    
    # Load the model
    print("Loading sentiment analysis model...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Model loaded!\n")
    
    print("="*60)
    print("STARTING ANALYSIS")
    print("="*60)
    
    all_results = []
    
    for idx, speech in df_filtered.iterrows():
        year = speech['Year']
        
        # Find president column (might be 'President' or 'president')
        president_col = 'President' if 'President' in df.columns else 'president'
        president = speech[president_col] if president_col in df.columns else speech[df.columns[0]]
        
        # Find title column
        title_col = 'Title' if 'Title' in df.columns else 'title'
        title = speech[title_col] if title_col in df.columns else speech[df.columns[2]]
        
        # Find text column
        text_col = 'Text' if 'Text' in df.columns else 'text'
        text = str(speech[text_col]) if text_col in df.columns else str(speech[df.columns[3]])
        
        print(f"\nAnalyzing {int(year)} - {president}")
        print(f"  Title: {title[:80]}")
        
        if text == 'nan' or text == '' or len(text) < 100:
            print(f"  ERROR: No valid text!")
            continue
        
        print(f"  Text length: {len(text)} characters")
        
        # Perform sentiment analysis
        analysis = analyze_speech_sentiment(text)
        
        result = {
            'year': int(year),
            'president': president,
            'title': title,
            'total_sentences': analysis['total_sentences'],
            'positive_count': analysis['positive_count'],
            'negative_count': analysis['negative_count'],
            'positive_percentage': round(analysis['positive_percentage'], 2),
            'negative_percentage': round(analysis['negative_percentage'], 2),
            'avg_positive_confidence': round(analysis['avg_positive_confidence'], 4),
            'avg_negative_confidence': round(analysis['avg_negative_confidence'], 4),
        }
        
        all_results.append(result)
        
        print(f"  ✓ Complete: {analysis['total_sentences']} sentences, "
              f"{analysis['positive_percentage']:.1f}% positive")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save to CSV
    output_filename = 'test_2010_2011_results.csv'
    results_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print(f"Analyzed {len(results_df)} speeches")
    print(f"Results saved to: {output_filename}")
    print("\nResults:")
    print(results_df[['year', 'president', 'positive_percentage', 'negative_percentage']])

if __name__ == "__main__":
    main()