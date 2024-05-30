import json
import csv
import time
import openai
import numpy as np
from tqdm import tqdm
import os
from argparse import ArgumentParser

openai.api_key = ""
openai.api_base = ""

MODEL = "gpt-3.5-turbo-1106"

def add_message(role, content, history):
    history.append({"role": role, "content": content})


def ai_request(history, t=0.2):
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=history,
        temperature=t,
    )
    output = response["choices"][0]["message"]["content"]
    return output

def baseline(dataset, start_index=0):
    print(f"Running baseline for dataset: {dataset}")
    # Create directory for logs if it doesn't exist
    log_dir = f'log/{dataset}'
    os.makedirs(log_dir, exist_ok=True)

    # Open the JSONL file
    with open(f'data/{dataset}/test.jsonl', 'r', encoding='utf-8') as file:
        # Read all lines from the file
        lines = file.readlines()

        for i, line in tqdm(enumerate(lines[start_index:], start=start_index)):
            # Parse the JSON data
            data = json.loads(line)

            # Extract data from the JSON object
            id = data['id']
            question = data['question']
            answers = data['answer'].split("#### ")[-1]

            prompt0 = f"""You are a knowledgeable and patient professor whose role is to guide students in solving problems correctly.
Here is a question:
{question}
Note: Since your responsibility is to guide students in answering the question, your analysis should think step by step, 
Please note that your role is to guide them step by step through the problem, so please don't give them the final result
"""
            teacher_history = []
            add_message('user', prompt0, teacher_history)
            teacher_output = ai_request(teacher_history)
            add_message('assistant', teacher_output, teacher_history)
            time.sleep(1)

            student1 = f"""To solve the problem, Please think and reason step by step, then answer.
Question:
{question}  
Generation Format:
Reasoning process:
Answer:
"""

            student_history = []
            add_message('user', student1, student_history)
            student_output1 = ai_request(student_history)
            add_message('assistant', student_output1, student_history)
            time.sleep(1)
            student_correctness = 'True' if answers.lower() in student_output1.lower() else 'False'

            student2 = f"""Task:
The question contains a large set of semi-synthetic stories involving hypothetical families. 
The task is to infer the relationship between two family members, whose relationship is not explicitly mentioned in the given story.
This is an credible analysis of this question:
{teacher_output}
Please verify your reasoning process for errors based on this analysis,
then refine your reasoning process and answer.
For question: How is [A] related to [B], your answer should be [A] is [B]'s [relationship].
Generation Format:
inference process:
Answer: 
"""
            add_message('user', student2, student_history)
            student_output2 = ai_request(student_history)
            add_message('assistant', student_output2, student_history)
            #guide_correctness = 'True' if any(answer.lower() in student_output2.lower() for answer in answers) else 'False'
            guide_correctness = 'True' if answers.lower() in student_output2.lower() else 'False'
            time.sleep(1)

            # Save log
            log_filename = f'{dataset}_{i}.json'
            log_path = os.path.join(log_dir, log_filename)
            with open(log_path, 'w') as log_file:
                json.dump({
                    "id": id,
                    "question": question,
                    'answer': answers,
                    "student_correctness": student_correctness,
                    "guide_correctness": guide_correctness,
                    "teacher_log": teacher_history,
                    "student_log": student_history,
                }, log_file, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(description="Baseline script with dataset and start index arguments")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--start_index", type=int, default=0, help="Start index to begin processing")
    args = parser.parse_args()

    baseline(args.dataset, args.start_index)


# python run.py --dataset CLUTRR --start_index 0