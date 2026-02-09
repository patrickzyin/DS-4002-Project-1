
from transformers import pipeline
import sys

def analyze_speech_sentiment(speech_text, chunk_size=512):
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    
    sentences = [s.strip() for s in speech_text.split('.') if s.strip()]
    
    print(f"Analyzing {len(sentences)} sentences...")
    
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
                print(f"Error analyzing sentence {i}: {e}")
    
    
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

def print_summary(analysis_results):
    """Print a summary of the sentiment analysis"""
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS SUMMARY")
    print("="*60)
    print(f"Total sentences analyzed: {analysis_results['total_sentences']}")
    print(f"\nPositive sentences: {analysis_results['positive_count']} ({analysis_results['positive_percentage']:.1f}%)")
    print(f"Negative sentences: {analysis_results['negative_count']} ({analysis_results['negative_percentage']:.1f}%)")
    print(f"\nAverage positive confidence: {analysis_results['avg_positive_confidence']:.3f}")
    print(f"Average negative confidence: {analysis_results['avg_negative_confidence']:.3f}")
    
    
    if analysis_results['positive_percentage'] > analysis_results['negative_percentage']:
        overall = "POSITIVE"
    elif analysis_results['negative_percentage'] > analysis_results['positive_percentage']:
        overall = "NEGATIVE"
    else:
        overall = "NEUTRAL"
    
    print(f"\nOverall speech sentiment: {overall}")
    print("="*60)
    
    
    sorted_results = sorted(
        analysis_results['detailed_results'],
        key=lambda x: x['score'],
        reverse=True
    )
    
    print("\nMost confident POSITIVE sentences:")
    positive_sentences = [r for r in sorted_results if r['label'] == 'POSITIVE'][:3]
    for i, result in enumerate(positive_sentences, 1):
        print(f"\n{i}. {result['sentence'][:100]}...")
        print(f"   Confidence: {result['score']:.3f}")
    
    print("\nMost confident NEGATIVE sentences:")
    negative_sentences = [r for r in sorted_results if r['label'] == 'NEGATIVE'][:3]
    for i, result in enumerate(negative_sentences, 1):
        print(f"\n{i}. {result['sentence'][:100]}...")
        print(f"   Confidence: {result['score']:.3f}")


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            speech_text = f.read()
    else:
        speech_text = """
        My fellow Americans, tonight I want to speak about the state of our union
        and share my vision for the future of our nation. We face great challenges,
        but I believe in the resilience and strength of the American people.
        Our economy is growing stronger each day. Unemployment is down and 
        businesses are thriving. However, we still have work to do to ensure
        that prosperity reaches every community. We must invest in education,
        infrastructure, and innovation. Together, we can build a brighter future
        for all Americans. The road ahead may be difficult, but I am confident
        that we will overcome any obstacle. Thank you, and God bless America.
        """
    
    results = analyze_speech_sentiment(speech_text)
    
    print_summary(results)

if __name__ == "__main__":
    main()