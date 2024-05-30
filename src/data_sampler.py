import json
import random

with open('data/SVAMP/test_origin.jsonl', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 随机采样500条数据
sampled_lines = random.sample(lines, 500)

# 对采样数据重新编排id并保存到新的列表中
sampled_data = []
for idx, line in enumerate(sampled_lines):
    data = json.loads(line)
    data['id'] = idx  # 重新编排id
    data['answer'] = data['answer'].split("#### ")[-1]
    sampled_data.append(data)

# 保存采样数据到新的JSONL文件
with open('data/SVAMP/test.jsonl', 'w', encoding='utf-8') as file:
    for data in sampled_data:
        file.write(json.dumps(data, ensure_ascii=False) + '\n')

