import os
import json
import csv
from argparse import ArgumentParser

# Initialize ArgumentParser
parser = ArgumentParser(description='Process dataset name.')
parser.add_argument('--dataset', metavar='dataset', type=str, help='Name of the dataset')
args = parser.parse_args()
dataset = args.dataset

# Define CSV file name and field names
csv_filename = f'log/baseline/{dataset}_extracted.csv'  # 修改保存路径
fieldnames = ['id', 'correctness', 'question', 'answer', 'log_content']

# Count the number of files in the directory
file_count = len(os.listdir(f'log/baseline/{dataset}'))

# Open the CSV file for writing
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Define the range of file names to process
    for i in range(file_count):
        file_path = f'log/baseline/{dataset}/{dataset}_{i}.json'  # Adjust the file path pattern as needed
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as jsonfile:
                    data = json.load(jsonfile)

                    # Extract the required fields
                    extracted_data = {
                        'id': data['id'],
                        'correctness': data['correctness'],
                        'question': data['question'],
                        'answer': data['answer'],
                        'log_content': data['log'][0]['content']
                    }

                    # Write to the CSV file
                    writer.writerow(extracted_data)
            except Exception as e:
                print("Error processing file:", file_path)
                print("Error message:", str(e))

print("Extraction complete and saved to", csv_filename)


