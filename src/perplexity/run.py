import argparse
import torch
from tqdm import tqdm
import models
import utils
import json
import pandas as pd
import numpy as np


def compute_perplexity_data(model, data_path):
    # For expedience, we're going to assume everything fits in memory for now
    # Also for expedience we're just going to save lists of arrays
    probs= np.array([])
    data = pd.read_json(data_path, lines=True)
    for i in tqdm(range(len(data))):
        output = model.get_perplexity_data(data["answer"][i])
        probs = np.append(probs, output['logprobs'])
    return probs


def main():
    model = models.create_model()
    data_path = "/data/llama-meta/data/data.jsonl"

    perplexity_data = compute_perplexity_data(
        model=model,
        data_path=data_path,
    )


    aggregate_logprobs = perplexity_data
    perplexity = np.exp(-aggregate_logprobs.mean())
    result = {
        "perplexity": float(perplexity)
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()