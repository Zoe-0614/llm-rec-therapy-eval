from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio") 

with open('training_input.txt', 'r') as file:
# with open('input.txt', 'r') as file:
    input_lines = file.readlines()


with open('output_mamba_1.txt', 'w') as outfile:
      for line in input_lines:
        line = line.strip()  
        if line:
            conversation = [{"role": "user", "content": line,}]
            completion = client.chat.completions.create(
                        messages=conversation,
                        model="hub",
                        temperature=0.5,
                      )
            output_text = completion.choices[0].message.content
            outfile.write(output_text + '\n*\n')    