import csv
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--dataset', required=True)
args = parser.parse_args()

dataset = args.dataset

correctness_columns = ['correctness']

def evaluate(file_name):
    with open(file_name, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        data = list(reader)

    for column in correctness_columns:
        column_accuracy = calculate_accuracy(data, column)
        print(f"{file_name} {column} Accuracy: {column_accuracy:.2f}%")

def calculate_accuracy(data, column):
    correctness_values = [row[column] for row in data]
    accuracy = 100 * np.mean([1 if correctness == 'True' else 0 for correctness in correctness_values])
    return accuracy


file_name1 = f'log/baseline/{dataset}_extracted.csv'

evaluate(file_name1)

