import re, ast, os, sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path


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


def process_directory(directory_path):
    """Loops through .txt files in the directory and processes each one."""
    num_runs = count_files_in_directory(directory_path)
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
            # llm_a_post = parse_post_conversation_scores(text, "LLM A")
            chatbot_scores[0] = llm_a_pre

            turn_counter = 1
            for turn, scores in parsed_data:
                if turn % 2 == 1:
                    chatbot_scores[turn_counter] = scores
                    turn_counter += 1

            # llm_b_pre = parse_pre_conversation_scores(text, "LLM B")
            # llm_b_post = parse_post_conversation_scores(text, "LLM B")

            results[run_idx] = chatbot_scores

    return results


# # Create subplots (5 rows, 1 column)
# fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

# # Iterate over each run and plot on the same subplots
# for run_idx in tqdm(range(99)):  # Iterate through 99 runs
#     data = results[run_idx]
#     time_points = sorted(data.keys())  # Extract and sort time points
    
#     for i, (trait, color) in enumerate(zip(traits, colors)):
#         trait_values = [data[t][trait] for t in time_points]  # Extract trait values
#         axes[i].plot(time_points, trait_values, linestyle='-', color=color, alpha=0.3)  # Line plot with transparency

#         # Formatting
#         axes[i].set_ylim(20, 50)
#         axes[i].set_ylabel(trait, color=color)
#         axes[i].tick_params(axis='y', colors=color)
#         axes[i].grid(True)
#         axes[i].legend([trait], loc="upper left")  # Single label for trait

# # Formatting
# axes[-1].set_xlabel("Time")
# plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout
# plt.savefig("Stacked_Evol_Line_Plot.png", bbox_inches='tight')  # Save final plot
# plt.show()

# # Define traits and colors
# traits = ['O', 'C', 'E', 'A', 'N']
# colors = ['r', 'g', 'b', 'm', 'c']  # Red, Green, Blue, Magenta, Cyan

# # Load results from directory
# directory_name = "Results/evol_outputs_LLM-A_mistralai_Mistral-Small-24B-Instruct-2501_LLM-B_gpt-4o-mini"
# results = process_directory(directory_name)  # Your function for loading files

# # Create subplots (5 rows, 1 column)
# fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)

# # Iterate over each run and plot deltas
# for run_idx in tqdm(range(99)):  # Iterate through 99 runs
#     data = results[run_idx]
#     time_points = sorted(data.keys())  # Extract and sort time points

#     # Extract initial values (timestep 0) for each trait
#     initial_values = {trait: data[time_points[0]][trait] for trait in traits}

#     for i, (trait, color) in enumerate(zip(traits, colors)):
#         # Compute deltas (difference from initial value)
#         deltas = [data[t][trait] - initial_values[trait] for t in time_points]
        
#         # Plot the delta values as a line plot
#         axes[i].plot(time_points, deltas, linestyle='-', color=color, alpha=0.3)

#         # Formatting
#         axes[i].set_ylabel(trait, color=color)
#         axes[i].tick_params(axis='y', colors=color)
#         axes[i].grid(True)
#         axes[i].axhline(y=0, color='black', linestyle='--', linewidth=0.8)  # Add a reference line at y=0
#         axes[i].legend([trait], loc="upper left")  # Single label for trait

# # Formatting
# axes[-1].set_xlabel("Time")
# plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout
# plt.savefig("Stacked_Delta_Evol_Plot.png", bbox_inches='tight')  # Save final plot
# plt.show()

# Define traits and colors
traits = ['E', 'A', 'C', 'N', 'O']
trait_names = ['E', 'A', 'C', 'ES', 'I']
colors = ['r', 'g', 'b', 'm', 'c']  # Red, Green, Blue, Magenta, Cyan

# Load results from directory
directory_name = "Results/evol_outputs_LLM-A_mistralai_Mistral-Small-24B-Instruct-2501_LLM-B_gpt-4o-mini"
results = process_directory(directory_name)  # Your function for loading files


# Initialize storage for averaged deltas
all_time_points = sorted(results[0].keys())  # Extract sorted time points, [0, 1, ..., 30]
num_time_steps = len(all_time_points) # len = 31

# Initialize dictionary to store deltas for each trait at each time step
delta_values = {trait: np.zeros(num_time_steps - 1) for trait in traits}  # No delta for t=0
delta_std_dev = {trait: np.zeros(num_time_steps - 1) for trait in traits}  # For standard deviation

# Compute deltas and average across runs
for i, trait in enumerate(traits):
    all_deltas = []  # Store all runs' deltas for this trait

    for run_idx in range(99):  # Iterate through 99 runs
        data = results[run_idx]

        # Compute deltas for this run
        deltas = [data[all_time_points[t]][trait] - data[all_time_points[t-1]][trait] for t in range(1, num_time_steps)]
        all_deltas.append(deltas)

    # Convert list to numpy array for easier averaging
    all_deltas = np.array(all_deltas)  # Shape: (99, num_time_steps - 1)

    # Compute mean and standard deviation across runs
    delta_values[trait] = np.mean(all_deltas, axis=0)
    delta_std_dev[trait] = np.std(all_deltas, axis=0)

# Create subplots (5 rows, 1 column)
fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
fig.suptitle("Average Turn-by-Turn Change in Personality Traits Over Time", fontsize=16)

# Plot averaged deltas with shaded confidence region
for i, (trait, color) in enumerate(zip(traits, colors)):
    time_points = all_time_points[1:]  # Since deltas start at t=1

    # Plot mean delta values
    axes[i].plot(time_points, delta_values[trait], linestyle='-', color=color, label=trait_names[i], linewidth=2)

    # Add standard deviation shading (confidence region)
    axes[i].fill_between(time_points, 
                         delta_values[trait] - delta_std_dev[trait], 
                         delta_values[trait] + delta_std_dev[trait], 
                         color=color, alpha=0.2)

    # Formatting
    axes[i].set_ylim(-5, 12)
    axes[i].set_ylabel(trait_names[i], color=color)
    axes[i].tick_params(axis='y', colors=color)
    axes[i].grid(True)
    axes[i].axhline(y=0, color='black', linestyle='--', linewidth=0.8)  # Baseline at y=0
    axes[i].legend(loc="upper left")  # Single label for trait

# Formatting
axes[-1].set_xlabel("Turn")
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout
plt.savefig("Averaged_Delta_Plot.png", bbox_inches='tight')  # Save final plot


# Create subplots (5 rows, 1 column)
fig, axes = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
fig.suptitle("Value of Personality Traits Over Time", fontsize=16)

# Iterate over each run and plot on the same subplots
for run_idx in tqdm(range(99)):  # Iterate through 99 runs
    data = results[run_idx]
    time_points = sorted(data.keys())  # Extract and sort time points
    
    for i, (trait, color) in enumerate(zip(traits, colors)):
        trait_values = [data[t][trait] for t in time_points]  # Extract trait values
        axes[i].plot(time_points, trait_values, linestyle='-', color=color, alpha=0.3)  # Line plot with transparency

        # Formatting
        axes[i].set_ylim(10, 50)
        axes[i].set_ylabel(trait_names[i], color=color)
        axes[i].tick_params(axis='y', colors=color)
        axes[i].grid(True)
        axes[i].legend([trait], loc="upper left")  # Single label for trait

# Formatting
axes[-1].set_xlabel("Turn")
plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust layout
plt.savefig("Stacked_Evol_Line_Plot.png", bbox_inches='tight')  # Save final plot
plt.show()