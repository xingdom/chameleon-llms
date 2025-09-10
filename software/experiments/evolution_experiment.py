import time, random, json, os, shutil, argparse
from data import *
from sys import exit
from tqdm import tqdm
from openai import OpenAI
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
        prompt = f"As someone who is '{personality_description}', " + prompt
    
    if context:
        prompt = f"Consider the following conversation you've just had:\n{context}\n\n" + prompt
    
    answer, prompt_tokens, completion_tokens = get_llm_response(prompt, llm_name)
    return question, trait, is_positive, answer, prompt_tokens, completion_tokens
    

# Conducts the BFPT personality test and returns scores per trait
def conduct_personality_test(llm_name, round_name, context=None, personality_description=None):

    global USED_PROMPT_TOKENS
    global USED_COMPLETION_TOKENS
    test_unknowns = 0

    # TODO remove, ablation study take context and replace all conversations with xxx
    if context:
        context = "x"*len(context)
    
    # Initialize the scores for each trait with their base values
    scores = base_scores.copy()

    L_out = []
    with ThreadPoolExecutor(max_workers=10) as executor:
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

            print(f"Test unknown for question: {question}")
            print(f"Received response: {answer}")
        
        USED_PROMPT_TOKENS += prompt_tokens
        USED_COMPLETION_TOKENS += completion_tokens
                
    if test_unknowns:
        print("Test unknowns:", test_unknowns)
    return scores, test_unknowns


def get_next_turn(llm_a_name, llm_b_name, convo_id, run_dict, turn_number):
    # If turn is even, user speaks, if turn is odd, chatbot speaks
    llm_b_personality = run_dict["llm_b_personality"],
    topic = run_dict["topic"]
    llm_a_role, llm_b_role = run_dict["llm_a_role"], run_dict["llm_b_role"]
    current_context = run_dict["context"]

    first_prompt_b = f"{llm_b_name}, {llm_b_role} Considering your personality of '{llm_b_personality}', start off the conversation naturally and realistically (in 50 words or less) with {llm_a_name} with this context:\n"

    prompt_b_template = f"{llm_b_name}, {llm_b_role} Considering your personality of '{llm_b_personality}', respond naturally and realistically (in 50 words or less) to {llm_a_name} with this context:\n"

    prompt_a_template = f"{llm_a_name}, {llm_a_role} Respond naturally and realistically (in 50 words or less) to {llm_b_name} with this context:\n"

    if turn_number == 0:
        # First prompt, user bot initiates conversation
        prompt = f"{first_prompt_b} {current_context}\n"
        llm_name = llm_b_name
    elif turn_number % 2 == 0:
        # User bot speaks
        prompt = f"{prompt_b_template} {current_context}\n"
        llm_name = llm_b_name

    else: 
        # Chatbot speaks
        prompt = f"{prompt_a_template} {current_context}\n"
        llm_name = llm_a_name

    response, prompt_tokens, completion_tokens = get_llm_response(prompt, llm_name)
    return response, convo_id, prompt_tokens, completion_tokens
    

def calculate_personality_shifts(before, after):
    shifts = {trait: after[trait] - before[trait] for trait in before}
    return shifts


def run_experiments(num_runs=10, num_turns=10, output_folder=None, sleep_time=0):

    # Load the averaged JSON file into a dictionary
    with open("base_same_model/averaged_results.json", "r", encoding="utf-8") as f:
        personality_score_dict = json.load(f)

    llm_a_name, llm_b_name = "LLM A", "LLM B"
    
    chosen_topics_and_roles = [random.choice(topics_with_roles) for _ in range(num_runs)]
    chosen_personalities_b = [random.choice(extreme_personality_tropes) for _ in range(num_runs)]
    

    # ---- Step 1: Pre-conversation personality profiling ----
    # NOTE LLM A is not given a personality, hence only one test is needed
    pre_conversation_score_a, _ = conduct_personality_test(llm_a_name, "Pre-conversation") 

    pre_conversation_scores_b = {}
    for idx, personality_b in enumerate(chosen_personalities_b):
        pre_conversation_scores_b[idx] = personality_score_dict[personality_b]

        # Write out first set of logs
        with open(f"{output_folder}/output_{idx}.txt", "w", encoding="utf-8") as f:
            f.write(f"{llm_b_name} is assigned the extreme personality: {chosen_personalities_b[idx]}\n")
            f.write(f"Pre-conversation personality scores for LLM A: {pre_conversation_score_a}\n")
            f.write(f"Pre-conversation personality scores for LLM B: {pre_conversation_scores_b[idx]}\n\n")

            topic, llm_a_role, llm_b_role = chosen_topics_and_roles[idx]
            f.write(f"B receives this prompt: {llm_b_name}, {llm_b_role} Considering your personality of '{chosen_personalities_b[idx]}', respond naturally and realistically (in 50 words or less) to {llm_a_name} with this context:\n\n")

            f.write(f"A receives this prompt: {llm_a_name}, {llm_a_role} Respond naturally and realistically (in 50 words or less) to {llm_b_name} with this context:\n\n")

            f.write(f"Conversation scenario: {topic}\n\n")

    # ---- Step 2: Run conversation ----
    # Initialize run information
    convo_tracker = {}
    for i in range(num_runs):
        topic, llm_a_role, llm_b_role = chosen_topics_and_roles[i]
        llm_b_personality = chosen_personalities_b[i]
        run_dict = {
            "convo_id": i,
            "llm_b_personality": llm_b_personality,
            "topic": topic,
            "llm_a_role": llm_a_role,
            "llm_b_role": llm_b_role,
            "context": "",
            "shifts_a": None,
            "shifts_b": None
        }
        convo_tracker[i] = run_dict

    # Run each turn in parallel. Next turn begins after all threads complete current turn
    for turn in range(num_turns):
        L_out = []
        with ThreadPoolExecutor(max_workers=50) as executor:
            future_to_item = {executor.submit(get_next_turn, llm_a_name, llm_b_name, run_dict["convo_id"], run_dict, turn): run_dict for run_dict in convo_tracker.values()}

            for future in tqdm(as_completed(future_to_item), total=len(future_to_item), desc=f"Turn {turn}"):
                result = future.result()
                if result:
                    L_out.append(result)

        llm_name = "LLM B" if turn % 2 == 0 else "LLM A"
        for response, convo_id, prompt_tokens, completion_tokens in tqdm(L_out, total=len(L_out), desc="Processing turn info"):
            convo_tracker[convo_id]["context"] += f"{llm_name}: {response}\n"

            # Log all responses in current turn
            with open(f"{output_folder}/output_{convo_id}.txt", "a", encoding="utf-8") as f:
                if turn % 2 == 0:
                    f.write(f"(Turn {turn}) {llm_b_name}: {response}\n")

                    # Get assessment scores for evol experiment
                    if turn % 20 == 0:
                        turn_conversation_scores_llmB, _ = conduct_personality_test(llm_b_name, f"Mid-conversation turn {turn}", context=convo_tracker[idx]["context"], personality_description=convo_tracker[idx]["llm_b_personality"])
                    else: 
                        turn_conversation_scores_llmB = {'E': 0, 'A': 0, 'C': 0, 'N': 0, 'O': 0}
                    f.write(f"(Turn {turn}) LLM B personality assessment: {turn_conversation_scores_llmB}")

                else:
                    f.write(f"(Turn {turn}) {llm_a_name}: {response}\n")
                    
                    # Get assessment scores for evol experiment
                    turn_conversation_scores_llmA, _ = conduct_personality_test(llm_a_name, f"Mid-conversation turn {turn}", context=convo_tracker[idx]["context"])

                    f.write(f"(Turn {turn}) LLM A personality assessment: {turn_conversation_scores_llmA}")
                
            global USED_PROMPT_TOKENS
            global USED_COMPLETION_TOKENS
            USED_PROMPT_TOKENS += prompt_tokens
            USED_COMPLETION_TOKENS += completion_tokens


    # Step 3: Post-conversation personality profiling with context
    post_conversation_scores_a, post_conversation_scores_b = {}, {}

    for idx in tqdm(range(num_runs), total=num_runs, desc="Post-conversation scores"):

        # Conduct personality tests
        post_conversation_scores_a[idx], _ = conduct_personality_test(llm_a_name, "Post-conversation", context=convo_tracker[idx]["context"])

        post_conversation_scores_b[idx], _ = conduct_personality_test(llm_b_name, "Post-conversation", context=convo_tracker[idx]["context"], personality_description=convo_tracker[idx]["llm_b_personality"])

        time.sleep(sleep_time)
        
    # Save personality shifts in convo_tracker
        shifts_a = calculate_personality_shifts(pre_conversation_score_a, post_conversation_scores_a[idx])
        shifts_b = calculate_personality_shifts(pre_conversation_scores_b[idx], post_conversation_scores_b[idx])

        convo_tracker[idx]["shifts_a"] = shifts_a
        convo_tracker[idx]["shifts_b"] = shifts_b

        with open(f"{output_folder}/output_{idx}.txt", "a", encoding="utf-8") as f:
            f.write(f"\nPost-conversation personality scores for LLM A: {post_conversation_scores_a[idx]}\n")
            f.write(f"Post-conversation personality scores for LLM B: {post_conversation_scores_b[idx]}\n")
            f.write(f"Personality shifts for LLM A after conversation with LLM B: {shifts_a}\n")
            f.write(f"Personality shifts for LLM B after conversation with LLM A: {shifts_b}\n\n")

            f.write(f"The following are: B's personality, chatbot's initial score, chatbot's shifts, user's initial score, and user's shifts.\n")
            f.write(f"{chosen_personalities_b[idx]}\n{pre_conversation_score_a}\n{shifts_a}\n{pre_conversation_scores_b[idx]}\n{shifts_b}\n")

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
            base_url=""
        )

    # TODO check API key
    openai_api_key = 'EMPTY'
    openai_client = OpenAI(
        api_key="", # TODO api key has been removed
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

    output_folder = f"DOT_evol_outputs_LLM-A_{LLM_A_MODEL_NAME.replace(r'/', '_')}_LLM-B_{LLM_B_MODEL_NAME.replace(r'/', '_')}"
    ensure_clean_folder(output_folder)

    # TODO check num_runs, num_turns, sleep_time (set to 0.5 if both models are OpenAI models). 
    start_time = time.time()
    convo_tracker = run_experiments(num_runs=args.num_runs, num_turns=args.num_turns, output_folder=output_folder, sleep_time=args.sleep_time)
    execution_time = time.time() - start_time

    print(convo_tracker[0])
    print(f"Execution time: {execution_time:.1f} seconds") 
    print(f"Used prompt tokens: {USED_PROMPT_TOKENS}")
    print(f"Used completion tokens: {USED_COMPLETION_TOKENS}")
