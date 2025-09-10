import time, random, json, os, shutil
from time import sleep
from data import *
from sys import exit
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

vectorizer = TfidfVectorizer()
trope_vectors = vectorizer.fit_transform(extreme_personality_tropes)
similarity_matrix = cosine_similarity(trope_vectors)

def get_similar_tropes(trope_idx, n=3, min_similarity=0.1):
    similarities = similarity_matrix[trope_idx]
    similar_indices = np.argsort(similarities)[::-1][1:]
    similar_indices = [idx for idx in similar_indices if similarities[idx] >= min_similarity]
    if len(similar_indices) > n:
        return random.sample(similar_indices, n)
    
    while len(similar_indices) < n:
        random_idx = random.randint(0, len(extreme_personality_tropes) - 1)
        if random_idx not in similar_indices and random_idx != trope_idx:
            similar_indices.append(random_idx)
                
    return similar_indices

def generate_question(convo, personality_description):
    correct_choice = random.randint(0,3)
    similar_trope_indices = get_similar_tropes(extreme_personality_tropes.index(personality_description))
    random.shuffle(similar_trope_indices)

    t = 0
    answer_choices = []
    for i in range(4):
        if i == correct_choice:
            answer_choices.append(personality_description)
        else:
            answer_choices.append(extreme_personality_tropes[similar_trope_indices[t]])
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

total_score = 0
lock = threading.Lock()

def worker(id, test):
    print(f"Thread {id} starting")

    score, test_unknowns = 0, 0
    starting_question = id * 50
    for i in range(starting_question, starting_question + 50):
        question, answer = test[i]
        response = get_llm_response(question)
        sleep(1)
        if response[0] not in answer_letters:
            test_unknowns += 1
            print(f"Thread {id} answered question {i} with unknown response: '{response}' instead of {answer_letters[answer]}")
        elif response[0] == answer_letters[answer]:
            score += 1
        else:
            print(f"Thread {id} answered question {i} with {response} instead of {answer_letters[answer]}")
    
    with lock:
        global total_score
        total_score += score
    print(f"Thread {id} finished with score {score}/50 and {test_unknowns} test_unknowns")

def conduct_test(num_runs):
    
    test = generate_test(num_runs=num_runs) #[(question, answer_int)]

    file_path = "similar_test_content.txt"
    with open(file_path, "w") as file:
        for item in range(len(test)):
            file.write(f"Q{item}: {test[item][0]}\nAnswer:{answer_letters[test[item][1]]}\n\n")

    threads = []
    for i in range(10):
        thread = threading.Thread(target=worker, args=(i, test))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    start_time = time.time()

    # TODO check num_runs
    conduct_test(num_runs=500)
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"Score: {total_score}/500 or {total_score/5:.2f}%\n")
    print(f"Execution time: {execution_time:.1f} seconds") 

