import json

# Load the dataset from the JSON file
# filename = 'Alexander_Street_shareGPT.json'

# with open(filename, 'r') as file:
#     dataset = json.load(file)

# with open ("input.txt", "w") as f1:
# # Iterate through the dataset and print the "input" field
#     for record in dataset:
#         f1.write((record['input']) + '\n')

# with open ("output_GT.txt", "w") as f2:
# # Iterate through the dataset and print the "input" field
#     for record in dataset:
#         f2.write((record['output']) + '\n')



input_file_path = 'llama-7b.jsonl'
output_file_path = 'llama-7b.txt'

with open(input_file_path, 'r') as jsonl_file, open(output_file_path, 'w') as output_file:
    for line in jsonl_file:
        # Parse the JSON object from the line
        json_obj = json.loads(line)
        
        if 'choices' in json_obj:
            for choice in json_obj['choices']:
                # Check if 'turns' key exists in the choice
                if 'turns' in choice:
                    # Write 'turns' content to the output file
                    output_file.write('\n'.join(choice['turns']) + '\n*\n')