from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def parse_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read().strip().split('\n*\n') 
    return content

def responses(question, answer_a, answer_b, answer_c, answer_d, answer_e):
    prompt = f"""
    User question: {question}
    Vicuna-7b: {answer_a}
    Alpaca-7b: {answer_b}
    Llama2-7b: {answer_c}
    Mamba-gpt: {answer_d}
    Mistral-7b: {answer_e}
    """
    return prompt

# Path to your question and answer files
answer_files_paths = ['vicuna-7b.txt', 'alpaca-7b.txt', 'llama2-7b.txt', 'output_mamba_baseline.txt', 'output_mistral7B_baseline.txt']

# Parse the question and answer files
with open('training_input.txt', 'r') as file:
    questions = file.readlines()

all_answers = [[] for _ in range(len(questions))]
for file_index, answer_file in enumerate(answer_files_paths):
    answers = parse_file(answer_file)
    for answer_index, answer in enumerate(answers):
        all_answers[answer_index].append(answer)

# Ensure each question has the corresponding answers from all answer files
assert all(len(answer_set) == len(answer_files_paths) for answer_set in all_answers), "Each question must have the corresponding number of answers."

# Evaluate each question with its corresponding set of answers
evaluations = []
for i, question in enumerate(questions):
    evaluations.append(responses(
        question,
        all_answers[i][0], # answer_a from vicuna-7b
        all_answers[i][1], # answer_b from alpaca-7b
        all_answers[i][2], # answer_c from llama-7b
        all_answers[i][3], # answer_d from mamba-gpt
        all_answers[i][4], # answer_e from mistral-7b
    ))

with open('GPT4.txt', 'w') as outfile:
      for line in evaluations:
        line = line.strip()  
        if line:
            conversation = [{"role": "user", "content": line,}]
            completion = client.chat.completions.create(
                        messages=conversation,
                        model="rombodawg_open_gpt4_8x7b_v0.2",
                        temperature=0.5,
                      )
            evaluation = completion.choices[0].message.content
            outfile.write(evaluation + '\n*\n')    


"""
Please act as an impartial judge and evaluate the quality of the responses provided by five AI personal psychiatrists to the client's complaint or question displayed below. Your evaluation should be based solely on the counselling strategy provided. During the evaluation process, the defined expression rules should also be appropriately considered.
    You cannot solely judge the quality based on "whether or not more advice or suggestions are given". Begin your evaluation by comparing the five responses to the question given. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. 
    Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. Provide scores out of 100 for each of the counselling strategies below for each answer A to E.
    # Expression Rule
    1.The AI psychiatrist' responses should not contain anything about the identity, such as 'as a psychologist' or 'as your psychiatrist'. 
    2.The AI psychiatrist's response should be more like natural human conversation. Simply listing a solution list would make the response appear overly mechanized.

    Counselling strategy: "Information", "Direct Guidance", "Approval and Reassurance", "Restatement & Reflection & Listening", "Interpretation", "Self-disclosure", "Obtain Relevant Information"

    The final output verdict should strictly follow the format below for the five AI personal psychiatrists after evaluating all the questions and its respective answers from the five models (provide ONLY the scores, replacing each '_' sign): 
    Model [model name] = Information: _% Direct Guidance: _% Approval and Reassurance: _% Restatement & Reflection & Listening: _% Interpretation: _% Self-disclosure: _% Obtain Relevant Information _%
"""