import json

# Initialize a structure to hold the highest values and corresponding file names
highest_metrics = {
    'sentiment': {'file': None, 'value': -float('inf')},
    'coherence': {'file': None, 'value': -float('inf')},
    'rouge_score': {'file': None, 'value': -float('inf')},
    'readability_score': {'file': None, 'value': -float('inf')}
}

# File names
file_names = []
for i in range(1, 16):
    file_names.append(f'compile_eval_mamba_{i}.json')

# Function to update the highest value found for a metric
def update_highest(highest_metrics, metric, value, file_name):
    if value > highest_metrics[metric]['value']:
        highest_metrics[metric]['value'] = value
        highest_metrics[metric]['file'] = file_name

# Iterate through each file and update the structure with the highest values found
for file_name in file_names:
    with open(file_name, 'r') as file:
        data = json.load(file)

        # Update highest value for each metric
        update_highest(highest_metrics, 'sentiment', data['sentiment']['compound'], file_name)
        update_highest(highest_metrics, 'coherence', data['coherence'], file_name)
        update_highest(highest_metrics, 'readability_score', data['readability_score'], file_name)

        # Special handling for rouge_score since it's nested and we might compare by 'f' score of rouge-1 for simplicity
        rouge_f_score = data['rouge_score']['rouge-1']['f']
        update_highest(highest_metrics, 'rouge_score', rouge_f_score, file_name)

# Display the file with the highest value for each metric
for metric, info in highest_metrics.items():
    print(f"{metric}: {info['file']} with value {info['value']}")
