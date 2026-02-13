from transformers import pipeline
import pandas as pd
import re
import ast
import csv as csv_module
import os


# Helper function to parse speech text
def parse_speech_text(text):
    # Converts list-formatted text to a single string
    # Some speeches are stored as ["sentence 1", "sentence 2"]
    try:
        speech_list = ast.literal_eval(text)
        if isinstance(speech_list, list):
            return ' '.join(speech_list)
    except:
        pass
    return text


# Helper function to clean speech text
def clean_speech_text(text):
    # Removes audience reactions like (applause) and [laughter] from the text
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\[(?!\'[^\]]*$)[^\]]*\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# Helper function to normalize titles
def normalize_title(title):
    # Removes "Part 2:", "Part 3:", etc. from titles
    normalized = re.sub(r'^Part\s+\d+[\s:‐\-–—]+', '', title, flags=re.IGNORECASE)
    return normalized.strip()


# Helper function to check if title is a continuation
def is_continuation(title):
    # Checks if a speech title indicates it's a continuation (Part 2)
    return bool(re.match(r'^Part\s+\d+', title, flags=re.IGNORECASE))


# Main function to analyze sentiment of a speech
def analyze_speech_sentiment(speech_text):
    # Analyzes the sentiment of a speech by examining each sentence
    # Returns a dictionary with statistics about positive and negative sentences
    
    # Parse and clean the text
    parsed_text = parse_speech_text(speech_text)
    cleaned_text = clean_speech_text(parsed_text)
    
    # Split into sentences
    sentences = [s.strip() for s in cleaned_text.split('.') if s.strip()]
    
    # Track results
    positive_count = 0
    negative_count = 0
    positive_scores = []
    negative_scores = []
    
    # Analyze each sentence
    for i, sentence in enumerate(sentences):
        if len(sentence) > 10:  # minimum 10 character limit
            try:
                # Get sentiment from the model
                result = sentiment_analyzer(sentence[:512])[0]
                
                if result['label'] == 'POSITIVE':
                    positive_count += 1
                    positive_scores.append(result['score'])
                else:
                    negative_count += 1
                    negative_scores.append(result['score'])
                    
            except Exception as e:
                print(f"      Error on sentence {i}: {e}")
    
    # Calculate percentages
    total = positive_count + negative_count
    positive_pct = (positive_count / total * 100) if total > 0 else 0
    negative_pct = (negative_count / total * 100) if total > 0 else 0
    
    # Calculate average confidence scores
    avg_pos_conf = sum(positive_scores) / len(positive_scores) if positive_scores else 0
    avg_neg_conf = sum(negative_scores) / len(negative_scores) if negative_scores else 0
    
    return {
        'total_sentences': total,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'positive_percentage': positive_pct,
        'negative_percentage': negative_pct,
        'avg_positive_confidence': avg_pos_conf,
        'avg_negative_confidence': avg_neg_conf,
        'positive_scores': positive_scores,
        'negative_scores': negative_scores
    }


# Helper function to combine results from multi-part speeches
def combine_sentiment_results(results_list):
    # Combines results from multiple parts of the same speech
    
    total_sentences = sum(r['total_sentences'] for r in results_list)
    positive_count = sum(r['positive_count'] for r in results_list)
    negative_count = sum(r['negative_count'] for r in results_list)
    
    # Combine all confidence scores
    all_positive_scores = []
    all_negative_scores = []
    for r in results_list:
        all_positive_scores.extend(r['positive_scores'])
        all_negative_scores.extend(r['negative_scores'])
    
    # Calculate combined statistics
    positive_pct = (positive_count / total_sentences * 100) if total_sentences > 0 else 0
    negative_pct = (negative_count / total_sentences * 100) if total_sentences > 0 else 0
    
    avg_pos_conf = sum(all_positive_scores) / len(all_positive_scores) if all_positive_scores else 0
    avg_neg_conf = sum(all_negative_scores) / len(all_negative_scores) if all_negative_scores else 0
    
    return {
        'total_sentences': total_sentences,
        'positive_count': positive_count,
        'negative_count': negative_count,
        'positive_percentage': positive_pct,
        'negative_percentage': negative_pct,
        'avg_positive_confidence': avg_pos_conf,
        'avg_negative_confidence': avg_neg_conf
    }


# Main analysis function
def main():
    global sentiment_analyzer
    
    import sys
    csv_module.field_size_limit(sys.maxsize)
    
    print("\nState of the Union Sentiment Analysis")
    print("Analyzing speeches from 1850 to 2025")
    print("Classification: >50% positive sentences = POSITIVE speech\n")
    
    data_path = os.path.join('..', 'DATA', 'SOTU_data_final.csv')
    output_path = os.path.join('..', 'OUTPUTS', 'sentiment_analysis_results.csv')
    
    print(f"Reading data from: {data_path}")
    
    # Read CSV with custom parser to handle formatting issues
    print("Loading speeches...")
    rows = []
    with open(data_path, 'r', encoding='utf-8') as f:
        csv_reader = csv_module.reader(f, quotechar='"', delimiter=',')
        header = next(csv_reader)
        
        for row in csv_reader:
            if len(row) >= len(header):
                rows.append(row[:len(header)])
            elif len(row) >= 3:
                while len(row) < len(header):
                    row.append('')
                rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(rows, columns=header)
    df.columns = [col.lstrip('\ufeff') for col in df.columns]
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    print(f"Loaded {len(df)} total speeches\n")
    
    # Filter for years 1850-2025
    df_filtered = df[(df['Year'] >= 1850) & (df['Year'] <= 2025)].copy()
    print(f"Analyzing {len(df_filtered)} speeches from 1850-2025")
    
    # Create grouping for multi-part speeches
    df_filtered['normalized_title'] = df_filtered['Title'].apply(normalize_title)
    df_filtered['is_continuation'] = df_filtered['Title'].apply(is_continuation)
    
    def create_group_key(row):
        if row['is_continuation']:
            return f"{int(row['Year'])}_{row['normalized_title']}"
        else:
            return f"{int(row['Year'])}_{row['Title']}"
    
    df_filtered['group_key'] = df_filtered.apply(create_group_key, axis=1)
    unique_groups = df_filtered['group_key'].unique()
    
    print(f"Found {len(unique_groups)} unique speeches (some multi-part speeches combined)\n")
    
    # Load sentiment analysis model
    print("Loading AI sentiment model (this may take a minute)...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    print("Model loaded successfully!\n")
    
    print("Starting analysis...\n")
    
    all_results = []
    
    # Analyze each unique speech
    for speech_num, group_key in enumerate(sorted(unique_groups), 1):
        speech_parts = df_filtered[df_filtered['group_key'] == group_key].sort_values('Title')
        
        # Get speech info
        first_speech = speech_parts.iloc[0]
        year = first_speech['Year']
        president = first_speech['President']
        
        if first_speech['is_continuation']:
            display_title = first_speech['normalized_title']
        else:
            display_title = first_speech['Title']
        
        # Get crisis info if available
        is_crisis = first_speech['is_crisis'] if 'is_crisis' in speech_parts.columns else None
        crisis_name = first_speech['crisis_name'] if 'crisis_name' in speech_parts.columns else None
        
        print(f"[{speech_num}/{len(unique_groups)}] {int(year)} - {president}")
        print(f"  {display_title[:70]}...")
        
        # If multi-part speech, show all parts
        if len(speech_parts) > 1:
            print(f"  Combining {len(speech_parts)} parts:")
            for idx, part in speech_parts.iterrows():
                print(f"    {part['Title'][:50]}")
        
        # Analyze each part
        part_results = []
        total_chars = 0
        
        for part_num, (idx, speech) in enumerate(speech_parts.iterrows(), 1):
            text = str(speech['Text'])
            
            if text == 'nan' or text == '' or len(text) < 100:
                print(f"  Warning: Part {part_num} has no valid text")
                continue
            
            total_chars += len(text)
            
            if len(speech_parts) > 1:
                print(f"    Processing part {part_num}... ({len(text)} characters)")
            
            # Run sentiment analysis
            analysis = analyze_speech_sentiment(text)
            part_results.append(analysis)
        
        if not part_results:
            print(f"  Error: No valid text found")
            continue
        
        # Combine results if multi-part speech
        if len(part_results) > 1:
            combined_analysis = combine_sentiment_results(part_results)
        else:
            combined_analysis = part_results[0]
        
        # Classify as POSITIVE or NEGATIVE based on >50% threshold
        overall_sentiment = 'POSITIVE' if combined_analysis['positive_percentage'] > 50 else 'NEGATIVE'
        
        # Store results
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
            'overall_sentiment': overall_sentiment,
            'num_parts': len(speech_parts),
            'total_characters': total_chars
        }
        
        if is_crisis is not None:
            result['is_crisis'] = is_crisis
        if crisis_name is not None:
            result['crisis_name'] = crisis_name
        
        all_results.append(result)
        
        print(f"  Result: {combined_analysis['total_sentences']} sentences, "
              f"{combined_analysis['positive_percentage']:.1f}% positive -> {overall_sentiment}\n")
    
    # Create results DataFrame
    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values('year').reset_index(drop=True)
    
    # Save to CSV
    results_df.to_csv(output_path, index=False)
    
    # Print summary
    print("\nAnalysis Complete!")
    print(f"Analyzed {len(results_df)} speeches from {results_df['year'].min()} to {results_df['year'].max()}")
    print(f"Results saved to: {output_path}")
    

if __name__ == "__main__":
    main()