import json
import csv
import time
import openai
import numpy as np
from tqdm import tqdm
import os
from argparse import ArgumentParser
import re
import ast

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
    log_dir = f'log/baseline/{dataset}'
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
            answers = data['answer']

            prompt0 = f"Question: {question} \nAnswer: "

            history = []
            add_message('user', prompt0, history)
            output = ai_request(history)
            add_message('assistant', output, history)
            time.sleep(1)
            correctness = 'True' if answers in output else 'False'

            # Save log
            log_filename = f'{dataset}_{i}.json'
            log_path = os.path.join(log_dir, log_filename)
            with open(log_path, 'w') as log_file:
                json.dump({
                    "id": id,
                    "question": question,
                    'answer': answers,
                    "correctness": correctness,
                    "log": history,
                }, log_file, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(description="Baseline script with dataset and start index arguments")
    parser.add_argument("--dataset", help="Dataset name")
    parser.add_argument("--start_index", type=int, default=0, help="Start index to begin processing")
    args = parser.parse_args()

    baseline(args.dataset, args.start_index)


