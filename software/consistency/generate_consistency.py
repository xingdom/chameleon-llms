import time, random, json, os, shutil
from time import sleep
from data import *
from sys import exit
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed


client = OpenAI(
    api_key="",
    base_url="https://api.openai.com/v1"
)

model_name = "gpt-4o-mini"

def get_llm_response(prompt):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.7,
        n=1
    )
    output = response.choices.pop().message.content
    #prompt_tokens = response.usage.prompt_tokens
    #completion_tokens = response.usage.completion_tokens
    return output

answer_letters = ['A','B','C','D']

def generate_question(convo, personality_description):
    correct_choice = random.randint(0,3)
    tropes = random.sample(extreme_personality_tropes, 4)

    t = 0
    answer_choices = []
    for i in range(4):
        if i == correct_choice:
            answer_choices.append(personality_description)
        else:
            if tropes[t] == personality_description:
                t += 1
            answer_choices.append(tropes[t])
            t += 1
    
    question = f"{convo}\n"
    question += f"Given this conversation between LLM A and LLM B, which of the following best describe LLM B's personality?\n"
    for j in range(4):
        question += f"{answer_letters[j]}) {answer_choices[j]}\n"
    question += f"Only respond with ONE of the following letters and nothing else: A, B, C, or D."
    
    return question, correct_choice
    
output_folders = ["outputs_LLM-A_gpt-4o_LLM-B_gpt-4o-mini","outputs_LLM-A_gpt-4o-mini_LLM-B_gpt-4o-mini","outputs_LLM-A_meta-llama_Meta-Llama-3.1-8B-Instruct_LLM-B_gpt-4o-mini","outputs_LLM-A_microsoft_phi-4_LLM-B_gpt-4o-mini","outputs_LLM-A_Qwen_Qwen2.5-7B-Instruct_LLM-B_gpt-4o-mini"]

def generate_test(num_runs):
    test = []
    for folder in range(5):
        i, added = 0, 0
        while added < (num_runs // 5):
            file_path = f"{output_folders[folder]}/output_{i}.txt"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 40:
                        convo = ''.join(lines[8:-12])
                        assigned_personality = lines[0][43:-1]
                        question, answer = generate_question(convo, assigned_personality)
                        test.append((question, answer))
                        added += 1
            i += 1
            if i >= 1000:
                break

    print("Convos gathered:", len(test))
    return test


def print_test(num_runs=50):
    
    test = generate_test(num_runs=num_runs) #[(question, answer_int)]

    file_path = "consistency_test.txt"
    with open(file_path, "w") as file:
        for item in range(len(test)):
            file.write(f"Q{item}: {test[item][0]}\n\n")

    file_path = "consistency_answer_key.txt"
    with open(file_path, "w") as file:
        for item in range(len(test)):
            file.write(f"Q{item} Answer:{answer_letters[test[item][1]]}\n")
