from datasets import load_dataset
import json

# def filter_long_convos(example):
#     return len(example["conversation"]) >= 20

# def filter_language(example):
#     return example["language"] == "English"

# dataset = load_dataset("allenai/WildChat-1M")
# dataset = dataset["train"]
# filtered_dataset = dataset.filter(filter_language).filter(filter_long_convos)
# print(filtered_dataset)
# output_file = "wildchat_filtered.json"
# filtered_dataset.to_json(output_file)

output_file = "wildchat_filtered.json"
data_files = {"train": output_file}
re_dataset = load_dataset("json", data_files=data_files, split="train")
print(re_dataset)
# first_example = re_dataset[0]
# print(first_example)

for example in re_dataset:
    conversation = example['conversation']
    for idx, turn in enumerate(conversation): 
        role = turn['role']
        content = turn['content']
        print(f"Role: {role}")
        print(content[:100])

    break
