import re, ast, os, sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

personality_groupings = {
    "Financial and Resource Extremes": ["Obsessively Frugal", "Overly Materialistic"],
    "Dishonesty and Manipulation": ["Pathologically Lying", "Overly Charismatic Manipulator"],
    "Unpredictability and Chaos": ["Chaotic and Unpredictable", "Compulsively Rebellious", "Recklessly Brave"],
    "Future vs. Past Orientation": ["Obsessively Future-Oriented", "Overly Nostalgic", "Chronically Nostalgic"],
    "Independence and Reliance Extremes": ["Excessively Self-Reliant", "Overly Dependent"],
    "Trust and Suspicion Extremes": ["Excessively Trusting", "Paranoid and Distrustful", "Extremely Gullible"],
    "Rule and Authority Issues": ["Compulsively Rebellious", "Pathologically Passive"],
    "Superstition vs. Skepticism": ["Extremely Superstitious", "Compulsively Skeptical"],
    "Emotional Extremes": ["Extremely Empathetic", "Overly Stoic", "Emotionally Volatile", "Pathologically Jealous"],
    "Social Behavior Extremes": [
        "Aloof and Detached", "Overly Dramatic", "Highly Argumentative", 
        "Extremely Passive-Aggressive", "Compulsively Gossiping", "Chronically Cynical"
    ],
    "Decision-Making Extremes": ["Pathologically Indecisive", "Chronically Indecisive", "Obsessively Controlling"],
    "Caution vs. Risk-Taking": ["Overly Cautious and Anxious", "Extremely Fatalistic", "Pathologically Curious"],
    "Honesty vs. Deception": ["Pathologically Honest", "Compulsively Honest"],
    "Perfectionism vs. Sloppiness": ["Pathologically Perfectionist", "Obsessively Perfectionistic"],
    "Time Management Extremes": ["Obsessively Punctual", "Chronically Late", "Chronically Impatient"],
    "Communication and Expression Extremes": ["Chronically Verbose", "Chronically Sarcastic", "Overly Pedantic", "Extremely Literal-Minded"],
    "Competitiveness and Cooperation Extremes": ["Pathologically Competitive", "Relentlessly Competitive"],
    "Perspective on Reality": ["Pathologically Optimistic", "Perpetually Pessimistic", "Extremely Idealistic", "Overly Idealistic"],
    "Memory and Awareness Extremes": ["Chronically Forgetful", "Excessively Curious"],
    "Criticism and Judgment Extremes": ["Compulsively Critical", "Hypercritical"],
    "Self-Control and Indulgence Extremes": ["Chronically Indulgent"],
    "Apologetic Behavior Extremes": ["Excessively Apologetic", "Compulsively Apologetic"],
    "Helping and Involvement Extremes": ["Compulsively Helpful"]
}

def parse_file_into_text(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        text = file.read()  # Read entire file content
    return text


def count_files_in_directory(directory):
    return sum(1 for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)))


def get_personality_type(text):
    pattern = r"LLM B is assigned the extreme personality: (.*?)\n"
    match = re.search(pattern, text)
    if match:
        return f"[{match.group(1)}]"  # Wrap the extracted description in brackets
    return None  # Return None if no match is found


def parse_mid_conversation_scores(text):
    pattern = r"\(Turn (\d+)\) LLM [AB] personality assessment: (\{.*?\})"

    matches = re.findall(pattern, text)
    results = []

    for turn, assessment in matches:
        try:
            assessment_dict = ast.literal_eval(assessment)  
            results.append((int(turn), assessment_dict))
        except (SyntaxError, ValueError):
            continue  
    return results


def parse_pre_conversation_scores(text, llm_name):
    assert llm_name in ["LLM A", "LLM B"]
    if llm_name[-1] == "A":
        pattern = r"Pre-conversation personality scores for LLM A: (\{.*?\})"
    else: 
        pattern = r"Pre-conversation personality scores for LLM B: (\{.*?\})"
    match = re.search(pattern, text)

    if match:
        assessment = ast.literal_eval(match.group(1))  # Safely convert string to dictionary
        return assessment
    

def parse_post_conversation_scores(text, llm_name):
    assert llm_name in ["LLM A", "LLM B"]
    if llm_name[-1] == "A":
        pattern = r"Post-conversation personality scores for LLM A: (\{.*?\})"
    else: 
        pattern = r"Post-conversation personality scores for LLM B: (\{.*?\})"
    match = re.search(pattern, text)

    if match:
        assessment = ast.literal_eval(match.group(1))  # Safely convert string to dictionary
        return assessment

# Pre-conversation personality scores for LLM A: {'E': 33, 'A': 42, 'C': 33, 'N': 24, 'O': 45}
# Pre-conversation personality scores for LLM B: {'E': 29.0, 'A': 49.75, 'C': 24.25, 'N': 38.25, 'O': 34.5}
# Post-conversation personality scores for LLM A: {'E': 36, 'A': 49, 'C': 47, 'N': 47, 'O': 31}
# Post-conversation personality scores for LLM B: {'E': 26, 'A': 50, 'C': 28, 'N': 30, 'O': 35}

def process_directory(directory_path):
    """Loops through .txt files in the directory and processes each one."""
    num_runs = count_files_in_directory(directory_path)
    # results = {idx: {'O': [], 'C': [], 'E': [], 'A': [], 'N': []} for idx in num_runs}
    results = {}
    for run_idx, filename in enumerate(os.listdir(directory_path)):

        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)

            text = parse_file_into_text(file_path)
            parsed_data = parse_mid_conversation_scores(text)
            num_turns = len(parsed_data)
            llm_b_personality = get_personality_type(text)

            user_scores, chatbot_scores = {}, {}
            llm_a_pre = parse_pre_conversation_scores(text, "LLM A")
            llm_a_post = parse_post_conversation_scores(text, "LLM A")
            chatbot_scores[0] = llm_a_pre
            chatbot_scores[num_turns+2] = llm_a_post # +2 instead of +1 to keep all user turns even

            for turn, scores in parsed_data:
                if turn % 20 == 0:
                    user_scores[turn+1] = scores
                elif turn % 2 == 0:
                    continue
                else:
                    chatbot_scores[turn+1] = scores

            llm_b_pre = parse_pre_conversation_scores(text, "LLM B")
            llm_b_post = parse_post_conversation_scores(text, "LLM B")
            user_scores[0] = llm_b_pre
            user_scores[num_turns+1] = llm_b_post

            results[run_idx] = chatbot_scores

    return results


def average_lst(lst):
    return round(sum(lst)/len(lst), 2)


def average_results(results):
    num_runs = len(results)
    num_turns = len(results[0])
    averaged_scores_per_turn = {turn_idx: {trait:0 for trait in "OCEAN"} for turn_idx in range(num_turns)}
    for turn_idx in tqdm(range(num_turns)):
        actual_turn_idx = turn_idx*2
        O_scores = []
        C_scores = []
        E_scores = []
        A_scores = []
        N_scores = []
        for run_idx in range(num_runs):
            turn_dict_for_run = results[run_idx][actual_turn_idx]
            O_scores.append(turn_dict_for_run['O'])
            C_scores.append(turn_dict_for_run['C'])
            E_scores.append(turn_dict_for_run['E'])
            A_scores.append(turn_dict_for_run['A'])
            N_scores.append(turn_dict_for_run['N'])
        averaged_scores_per_turn[turn_idx]['O'] = average_lst(O_scores)
        averaged_scores_per_turn[turn_idx]['C'] = average_lst(C_scores)
        averaged_scores_per_turn[turn_idx]['E'] = average_lst(E_scores)
        averaged_scores_per_turn[turn_idx]['A'] = average_lst(A_scores)
        averaged_scores_per_turn[turn_idx]['N'] = average_lst(N_scores)

    print(averaged_scores_per_turn)
    return averaged_scores_per_turn

def get_personality_cluster_assignment(directory_path):
    p_group_assignment = []
    for run_idx, filename in enumerate(os.listdir(directory_path)):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            text = parse_file_into_text(file_path)
            text = text.split("\n")[0].split(":")[1].strip()
            for personality_group, personalities_in_group in personality_groupings.items():
                if text in personalities_in_group:
                    p_group_assignment.append(personality_group)
                    break
    return p_group_assignment


if __name__ == "__main__":
    print("Current Directory:", Path.cwd())
    # model_output_name = "evol_outputs_LLM-A_meta-llama_Meta-Llama-3.1-8B-Instruct_LLM-B_gpt-4o-mini"
    model_output_name = "evol_outputs_LLM-A_mistralai_Mistral-Small-24B-Instruct-2501_LLM-B_gpt-4o-mini"
    directory_name = f"Results/{model_output_name}"

    p_group_assignment = get_personality_cluster_assignment(directory_name)
    results_by_group = {key:[] for key in personality_groupings.keys()}
    results = process_directory(directory_name)

    for run_idx in range(len(p_group_assignment)):
        results_by_group[p_group_assignment[run_idx]].append(results[run_idx])
    

    for group_name, results in results_by_group.items():
        if len(results) == 0:
            continue

        averaged_scores_per_turn = average_results(results)
        data = averaged_scores_per_turn

        # Sort time points (x-axis)
        time_points = sorted(data.keys())

        # Extract values for each personality trait
        traits = ['O', 'C', 'E', 'A', 'N']
        trait_values = {trait: [data[t][trait] for t in time_points] for trait in traits}
        # Define colors for each trait
        colors = ['r', 'g', 'b', 'm', 'c']  # Red, Green, Blue, Magenta, Cyan

        # Create subplots (5 rows, 1 column)
        fig, axes = plt.subplots(5, 1, figsize=(6, 12), sharex=True)

        # Plot each trait in its own subplot with a unique color
        for i, (trait, color) in enumerate(zip(traits, colors)):
            axes[i].plot(time_points, trait_values[trait], marker='o', linestyle='-', color=color, label=trait)
            axes[i].set_ylim(20, 50)
            axes[i].set_ylabel(trait, color=color)  # Set label color to match line color
            axes[i].tick_params(axis='y', colors=color)  # Change tick color to match line
            axes[i].grid(True)
            axes[i].legend(loc="upper left")  # Add legend

        # Formatting
        axes[-1].set_xlabel("Time")  # Label x-axis on the last subplot
        fig.suptitle(f"Personality Trait Changes Over Time: all personalities")

        # Adjust layout and show plot
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust spacing to fit title
        plt.savefig(f'Mistral_Evol_Plots/{group_name}.png', bbox_inches='tight')