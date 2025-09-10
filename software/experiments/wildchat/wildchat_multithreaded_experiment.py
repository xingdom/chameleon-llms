import time, random, json, os, shutil, argparse
from data import *
from sys import exit
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from random import shuffle
from concurrent.futures import ThreadPoolExecutor, as_completed


def ensure_clean_folder(directory):
    """Ensures a directory exists and is empty."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


def get_llm_response(prompt, curr_llm_name):
    assert curr_llm_name in ["LLM A", "LLM B"]
    client = LLM_A_CLIENT if curr_llm_name == "LLM A" else LLM_B_CLIENT
    model_name = LLM_A_MODEL_NAME if curr_llm_name == "LLM A" else LLM_B_MODEL_NAME

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
    prompt_tokens = response.usage.prompt_tokens
    completion_tokens = response.usage.completion_tokens
    return output, prompt_tokens, completion_tokens


def answer_personality_question(personality_description, context, llm_name, question, trait, is_positive):
    prompt = f"{llm_name}, based on your personality, choose one of the following options for the statement:\n"
    prompt += f"'{question}'\n"
    prompt += f"A) Disagree\nB) Slightly Disagree\nC) Neutral\nD) Slightly Agree\nE) Agree\nOnly respond with ONE of the following letters and nothing else: A, B, C, D, or E."
    
    if personality_description:
        prompt = f"As someone who has said the following: '{personality_description}', " + prompt
    
    if context:
        prompt = f"Consider the following conversation you've just had:\n{context}\n\n" + prompt
    
    answer, prompt_tokens, completion_tokens = get_llm_response(prompt, llm_name)
    return question, trait, is_positive, answer, prompt_tokens, completion_tokens
    

# Conducts the BFPT personality test and returns scores per trait
def conduct_personality_test(llm_name, round_name, context=None, personality_description=None):
    global USED_PROMPT_TOKENS
    global USED_COMPLETION_TOKENS
    test_unknowns = 0
    
    # Initialize the scores for each trait with their base values
    scores = base_scores.copy()

    L_out = []
    with ThreadPoolExecutor(max_workers=50) as executor:
        future_to_item = {executor.submit(answer_personality_question, personality_description, context, llm_name, *info): info for info in questions}

        # for future in (as_completed(future_to_item), total=len(questions)):
        for future in as_completed(future_to_item):
            result = future.result()
            if result:
                L_out.append(result)
    
    
    for question, trait, is_positive, answer, prompt_tokens, completion_tokens in L_out:
        if answer[0] in pos_grading_scheme:
            if is_positive:
                scores[trait] += pos_grading_scheme[answer[0]]
            else:
                scores[trait] += neg_grading_scheme[answer[0]]
        else:
            test_unknowns += 1
            scores[trait] += 3
        
        USED_PROMPT_TOKENS += prompt_tokens
        USED_COMPLETION_TOKENS += completion_tokens
                
    if test_unknowns:
        print("Test unknowns:", test_unknowns)
    return scores, test_unknowns


def calculate_personality_shifts(before, after):
    shifts = {trait: after[trait] - before[trait] for trait in before}
    return shifts


def run_experiments(num_runs=1000, num_turns=20, output_folder=None, sleep_time=0):
    # Initialize run information by loading wildchat dataset
    llm_a_name, llm_b_name = "LLM A", "LLM B"

    # ---- Step 0: Process file ----
    wildchat_file = "base_same_model/wildchat_filtered.json"
    filtered_dataset = load_dataset("json", data_files={"train": wildchat_file}, split="train").shuffle()
    convo_tracker = {}
    for i_run in tqdm(range(num_runs), total=num_runs, desc="Formatting convo_tracker from wildchat data"):
        wildchat_convo = filtered_dataset[i_run]['conversation']
        run_dict = {
            "convo_id": i_run,
            "context": "",
            "user_utterances": "",
            "shifts_a": None,
            "shifts_b": None
        }

        context = ""
        user_utterances = ""
        for idx, turn in enumerate(wildchat_convo):
            if idx % 2 == 0:
                assert turn['role'] == "user"
            else:
                assert turn['role'] == "assistant"
            if idx >= num_turns:
                break
            turn_content = turn['content']
            context += f"{llm_b_name if idx % 2 == 0 else llm_a_name}: {turn_content}\n"

            with open(f"{output_folder}/output_{i_run}.txt", "a", encoding="utf-8") as f:
                if idx % 2 == 0:
                    f.write(f"(Turn {idx}) {llm_b_name}: {turn_content}\n")
                    user_utterances += f"Utterance {idx}: {turn['content']}\n"
                else:
                    f.write(f"(Turn {idx}) {llm_a_name}: {turn_content}\n")

        run_dict["context"] = context
        run_dict["user_utterances"] = user_utterances
        convo_tracker[i_run] = run_dict


    # ---- Step 1: Pre-conversation personality profiling ----
    pre_conversation_score_a, _ = conduct_personality_test(llm_a_name, "Pre-conversation") 
    pre_conversation_scores_b = {}
    for idx in tqdm(range(num_runs), total=num_runs, desc="Running pre scores for llm B"):
        pre_conversation_score_b, _ = conduct_personality_test(llm_b_name, "Pre-conversation", personality_description=convo_tracker[idx]["user_utterances"])
        pre_conversation_scores_b[idx] = pre_conversation_score_b
         
        with open(f"{output_folder}/output_{idx}.txt", "w", encoding="utf-8") as f:
            f.write(f"Pre-conversation personality scores for LLM A: {pre_conversation_score_a}\n")
            f.write(f"Pre-conversation personality scores for LLM B: {pre_conversation_scores_b[idx]}\n\n")


    # ---- Step 3: Post-conversation personality profiling with context ----
    post_conversation_scores_a = {}
    for idx in tqdm(range(num_runs), total=num_runs, desc="Post-conversation scores"):

        # Conduct personality tests
        post_conversation_scores_a[idx], _ = conduct_personality_test(llm_a_name, "Post-conversation", context=convo_tracker[idx]["context"])

        time.sleep(sleep_time)
        

    # Save personality shifts in convo_tracker
        shifts_a = calculate_personality_shifts(pre_conversation_score_a, post_conversation_scores_a[idx])
        convo_tracker[idx]["shifts_a"] = shifts_a

        with open(f"{output_folder}/output_{idx}.txt", "a", encoding="utf-8") as f:
            f.write(f"\nPost-conversation personality scores for LLM A: {post_conversation_scores_a[idx]}\n")

            f.write(f"The following are: chatbot initial score, chatbot's shifts, user initial score, copy of user initial score\n")
            f.write(f"{pre_conversation_score_a}\n{shifts_a}\n")
            f.write(f"{pre_conversation_scores_b[idx]}\n{pre_conversation_scores_b[idx]}\n")

    return convo_tracker


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Parse CLI arguments")
    parser.add_argument("--model_name", type=str, help="Name of model (string)")
    parser.add_argument("--model_port", type=int, default=0000, help="Port model is hosted on (int). Irrelevant if not using local model")
    parser.add_argument("--num_runs", type=int, default=1000, help="Number of runs for experiment. Each run is a topic and personality combination.")
    parser.add_argument("--num_turns", type=int, default=20, help="Number of turns for conversation. Each back-and-forth counts as two turns.")
    parser.add_argument("--sleep_time", type=float, default=0, help="")
    args = parser.parse_args()

    local_client = OpenAI(
            api_key="EMPTY",
            base_url=f"http://152.2.134.51:{args.model_port}/v1",
        )

    # TODO check API key
    openai_client = OpenAI(
        api_key="",
        base_url="https://api.openai.com/v1"
    )

    # TODO change LLM_A_CLIENT accordingly
    LLM_A_CLIENT = openai_client if "gpt" in args.model_name else local_client
    LLM_A_MODEL_NAME = args.model_name

    LLM_B_CLIENT = openai_client
    LLM_B_MODEL_NAME = "gpt-4o-mini"


    USED_PROMPT_TOKENS = 0
    USED_COMPLETION_TOKENS = 0

    # Base scores for each trait as per the scoring sheet
    # Extroversion, Agreeableness, Conscientiousness, Emotional Stability, Openness to Experience
    base_scores = {"E": 0, "A": 0, "C": 0, "N": 0, "O": 0}
    # Grading scheme (A-E responses for each LLM response)
    pos_grading_scheme = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
    neg_grading_scheme = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}

    output_folder = f"TEST_wildchat_outputs_LLM-A_{LLM_A_MODEL_NAME.replace(r'/', '_')}_LLM-B_{LLM_B_MODEL_NAME.replace(r'/', '_')}"
    ensure_clean_folder(output_folder)

    # TODO check num_runs, num_turns, sleep_time (set to 0.5 if both models are OpenAI models). 
    start_time = time.time()
    convo_tracker = run_experiments(num_runs=args.num_runs, num_turns=args.num_turns, output_folder=output_folder, sleep_time=args.sleep_time)
    execution_time = time.time() - start_time

    print(convo_tracker[0])
    print(f"Execution time: {execution_time:.1f} seconds") 
    print(f"Used prompt tokens: {USED_PROMPT_TOKENS}")
    print(f"Used completion tokens: {USED_COMPLETION_TOKENS}")
