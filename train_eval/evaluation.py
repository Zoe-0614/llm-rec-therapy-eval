from nltk.translate.bleu_score import SmoothingFunction 
from rouge import Rouge 
import textstat
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
import os
import json

# Initialize NLTK Sentiment Intensity Analyzer
sia = SentimentIntensityAnalyzer()

# Initialize the sentence transformer model for coherence score
model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_rouge_scores(reference, candidate):
    rouge = Rouge()
    scores = rouge.get_scores(candidate, reference)
    return scores


def calculate_readability(text):
    score = textstat.flesch_reading_ease(text)
    return score

def read_GT(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip().split('\n\n') 
    return content
    
def read_outputs(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip().split('\n*\n') 
    return content

def analyze_sentiment(text):
    score = sia.polarity_scores(text)
    return score

def calculate_coherence(sentences):
    embeddings = model.encode(sentences)
    coherence_scores = []
    for i in range(len(embeddings) - 1):
        score = 1 - cosine(embeddings[i], embeddings[i+1])
        coherence_scores.append(score)
    return np.mean(coherence_scores)

def evaluate_outputs(outputs, ground_truth):
    results = []
    for i in range(len(outputs)):
        sentences = nltk.sent_tokenize(outputs[i])
        sentiment_score = analyze_sentiment(outputs[i])
        coherence_score = calculate_coherence(sentences)
        sentences_GT = nltk.sent_tokenize(ground_truth[i])
        sentiment_score_GT = analyze_sentiment(ground_truth[i])
        coherence_score_GT = calculate_coherence(sentences_GT)
        rouge_scores = calculate_rouge_scores(ground_truth[i], outputs[i])
        readability_score = calculate_readability(outputs[i])
        
        results.append({
            'sentiment': sentiment_score,
            'coherence': coherence_score,
            'sentimentGT': sentiment_score_GT,
            'coherenceGT': coherence_score_GT,
            'rouge_score': rouge_scores,
            'readability_score': readability_score
        })
    return results


outputs = read_outputs(f'output_mistral7B_baseline.txt')
GT = read_GT('training_output_GT.txt')
evaluation_results = evaluate_outputs(outputs, GT)

# Initialize accumulators
totals = {
    'sentiment': {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0},
    'coherence': 0,
    'sentimentGT': {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0},
    'coherenceGT': 0,
    'rouge_score': {'rouge-1': {'r': 0, 'p': 0, 'f': 0}, 'rouge-2': {'r': 0, 'p': 0, 'f': 0}, 'rouge-l': {'r': 0, 'p': 0, 'f': 0}},
    'readability_score': 0,
}

# Process each entry
for entry in evaluation_results:
    for key in ['neg', 'neu', 'pos', 'compound']:
        totals['sentiment'][key] += entry['sentiment'][key]
        totals['sentimentGT'][key] += entry['sentimentGT'][key]
    totals['coherence'] += entry['coherence']
    totals['coherenceGT'] += entry['coherenceGT']
    totals['readability_score'] += entry['readability_score']
    for rouge_key in ['rouge-1', 'rouge-2', 'rouge-l']:
        for score_key in ['r', 'p', 'f']:
            totals['rouge_score'][rouge_key][score_key] += entry['rouge_score'][0][rouge_key][score_key]

# Calculate averages
averages = totals
num_entries = len(evaluation_results)
for key in ['neg', 'neu', 'pos', 'compound']:
    averages['sentiment'][key] /= num_entries
    averages['sentimentGT'][key] /= num_entries
averages['coherence'] /= num_entries
averages['coherenceGT'] /= num_entries
averages['readability_score'] /= num_entries
for rouge_key in ['rouge-1', 'rouge-2', 'rouge-l']:
    for score_key in ['r', 'p', 'f']:
        averages['rouge_score'][rouge_key][score_key] /= num_entries


# Write averages to a file
with open(f'compile_eval_mistral7B_baseline.json', 'w') as f:
    json.dump(averages, f, indent=4)

print("Averages calculated and written to .json")

