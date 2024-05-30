
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
import json
from llama import Llama
import pandas as pd
from tqdm import tqdm

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    dataset: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_seq_len: int = 1024,
    max_gen_len: int = 128,
    max_batch_size: int = 8,
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )
    new_rows_list=[]
    data = pd.read_json(f"data/{dataset}/test.jsonl", lines=True)
    output = f"log/{dataset}.jsonl"
    print(output)

    with open(output, 'w', encoding='utf-8') as file:
        file.write('') 

    data['input']=""
    for i in range(len(data)):
        data.loc[data.index[i],'input'] =  "Question:" + data['question'][i] + "\n" "Answer:"

    for i in tqdm(range(0, len(data), max_batch_size)):
        batch = data.iloc[i:i+max_batch_size]

        results = generator.text_completion(
            batch['input'],
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )
        
        for index, result in zip(batch.to_dict(orient='records'), results):
            new_rows_list.append({
                'id': index['id'],
                'baseline_answer': result['generation'],
                'ground_truth': index["answers"]
            })

        # Write the current batch to the JSON Lines file
        with open(output, 'a', encoding='utf-8') as jsonl_file:
            for row in new_rows_list:
                jsonl_file.write(json.dumps(row) + '\n')
                new_rows_list=[]

    n = 0
    x = 0
    # Read JSON Lines file and create a DataFrame
    with open(output, 'r', encoding='utf-8') as jsonl_file:
            for line in jsonl_file:
                n+=1
                data = json.loads(line)
                ground_truth = data['ground_truth']
                answers = str(data['baseline_answer'])
                if any(answer.lower() in answers.lower() for answer in ground_truth):
                    x+=1

    print(x/n)

if __name__ == "__main__":
    fire.Fire(main)

"""
python baseline.py \
    --ckpt_dir /data/LLaMA-2/meta/Llama-2-7b-meta \
    --tokenizer_path /data/LLaMA-2/meta/Llama-2-7b-meta/tokenizer.model  \
    --dataset date --max_seq_len 1024 --max_batch_size 8
"""
