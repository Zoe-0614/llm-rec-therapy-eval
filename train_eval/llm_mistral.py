from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")


with open('training_input.txt', 'r') as file:
    input_lines = file.readlines()


with open('output_mistral7B_baseline.txt', 'w') as outfile:
      for line in input_lines:
        line = line.strip()  
        if line:
            conversation = [{"role": "user", "content": line,}]
            completion = client.chat.completions.create(
                        messages=conversation,
                        model="mistralai_mistral-7b-instruct-v0.2",
                        temperature=0.5,
                      )
            output_text = completion.choices[0].message.content
            outfile.write(output_text + '\n*\n')    