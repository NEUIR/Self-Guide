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


def baseline(dataset, method, start_index=0):
    print(f"Running baseline for dataset: {dataset}")
    # Create directory for logs if it doesn't exist
    log_dir = f'log/{method}/{dataset}'
    os.makedirs(log_dir, exist_ok=True)

    # Open the JSONL file
    with open(f"log_guideline/{dataset}.jsonl", 'r', encoding='utf-8') as file:
        # Read all lines from the file
        lines = file.readlines()

        for i, line in tqdm(enumerate(lines[start_index:], start=start_index)):
            # Parse the JSON data
            data = json.loads(line)

            # Extract data from the JSON object
            id = data['id']
            question = data['question']
            answers = data['answer']
            guideline = data['guideline']

            #mmlu
            choice = data['choices']
            options = '\n'.join([f"{key}: {value}" for key, value in choice.items()])
            mcq = f"Question: {question}\n{options}"
            prompt0 = mcq + '\n' + "Choice: "

            # CLUTRR
            # prompt0 = question + '\n' + "Answer: "

            # # sqa
            # prompt0 = "Question: " + question + '\n' + "Your answer should be Yes or No. \nAnswer: "

            # date
            prompt_sd = mcq + '\n' + "Choice: "

            prompt_cot = f"""To solve the problem, Please think and reason step by step, then answer.
            {mcq} 
            Choice: """

            if method == "sd_verify":
                prompt0 = prompt_sd
            elif method == "cot_verify":
                prompt0 = prompt_cot
            else:
                raise ValueError("Invalid method. Method must be 'sd_verify' or 'cot_verify'.")

            history = []
            add_message('user', prompt0, history)
            output0 = ai_request(history)
            add_message('assistant', output0, history)
            time.sleep(1)

            # #date
            # prompt1 = "Can you double check that your answer is correct? Put your final answer in the form of MM/DD/YYYY."

            #CLUTRR
            # prompt1 = "Can you double check that your answer is correct? Please give your final answer."

            # sqa
            # prompt1 = "Can you double check that your answer is correct? Put your final answer in the form of Yes or No."

            # mmlu
            prompt1 = "Can you double check that your answer is correct? Please give your final answer. You should only output the letters that represent the answer."


            add_message('user', prompt1, history)
            output = ai_request(history)
            add_message('assistant', output, history)
            time.sleep(1)

            # mmlu
            ground_truth = answers.lower()
            theoutput = str(output).split()
            text_str = str(theoutput[0]).lower()
            print(ground_truth)
            print(text_str)

            correctness = 'True' if ground_truth in text_str else 'False'

            #sqa
            # first_words = ' '.join(output.split()[:1])
            # corrected_result = first_words.lower()  # 小写化结果
            # if 'yes' in corrected_result:
            #     correctness = True
            # elif 'no' in corrected_result:
            #     correctness = False
            # else:
            #     correctness = None  # 如果既不是'yes'也不是'no'，则将 correctness 设为 None

            # date/CLUTRR
            correctness = 'True' if answers.lower() in output.lower() else 'False'

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
    parser.add_argument("--method", help="sd_verify or cot_verify")
    args = parser.parse_args()

    baseline(args.dataset, args.method, args.start_index)


# python self_verify.py --dataset mmlu --start_index 0 --method sd_verify


# These are the solutions to the problem from other agents:
# <other agent responses>Using the reasoning from other agents as additional
# advice, can you give an updated answer? Examine your solution and that other
# agents

