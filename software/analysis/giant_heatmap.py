import os
import re
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def extract_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 4:
        return None
    
    data = []
    for line in lines[-4:]:
        try:
            # Find dictionary-like structure
            match = re.search(r'\{.*\}', line)
            if match:
                d = eval(match.group())
                if all(key in d for key in ['E', 'A', 'C', 'N', 'O']):
                    data.append(d)
                else:
                    return None
            else:
                return None
        except:
            return None
    
    return data if len(data) == 4 else None

def calculate_correlation(x, y):
    r, p = stats.pearsonr(x, y) #Pearson correlation = how strong is the LINEARITY
    slope, _, _, _, _ = stats.linregress(x, y)
    n = len(x)
    r_z = np.arctanh(r) #Applying Fisher's z-transformation bc the sampling distribution of r isn't normal
    se = 1/np.sqrt(n-3)
    z = stats.norm.ppf((1+0.95)/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z)) #Transform bounds back to r-space using inverse of Fisher's transformation
    return r, (lo, hi), slope


#output_folders = {"outputs_LLM-A_gpt-4o_LLM-B_gpt-4o-mini":"GPT-4o","outputs_LLM-A_gpt-4o-mini_LLM-B_gpt-4o-mini":"GPT-4o-mini","outputs_LLM-A_meta-llama_Meta-Llama-3.1-8B-Instruct_LLM-B_gpt-4o-mini":"Llama-8b","outputs_LLM-A_microsoft_phi-4_LLM-B_gpt-4o-mini":"Phi-4","outputs_LLM-A_Qwen_Qwen2.5-7B-Instruct_LLM-B_gpt-4o-mini":"Qwen-7b","outputs_LLM-A_mistralai_Mistral-Small-24B-Instruct-2501_LLM-B_gpt-4o-mini":"Mistral-24b","outputs_LLM-A_google_gemma-2-2b-it_LLM-B_gpt-4o-mini":"Gemma-2b"}

#6 heatmaps
output_folders = {"outputs_LLM-A_gpt-4o_LLM-B_gpt-4o-mini":"GPT-4o","outputs_LLM-A_gpt-4o-mini_LLM-B_gpt-4o-mini":"GPT-4o-mini","outputs_LLM-A_meta-llama_Meta-Llama-3.1-8B-Instruct_LLM-B_gpt-4o-mini":"Llama-8b","outputs_LLM-A_microsoft_phi-4_LLM-B_gpt-4o-mini":"Phi-4","outputs_LLM-A_mistralai_Mistral-Small-24B-Instruct-2501_LLM-B_gpt-4o-mini":"Mistral-24b","outputs_LLM-A_google_gemma-2-2b-it_LLM-B_gpt-4o-mini":"Gemma-2b"}

#output_folders = {"wildchat_outputs": "Wildchat_gpt4omini"}

#output_folders = {"outputs_LLM-A_microsoft_phi-4_LLM-B_gpt-4o-mini":"MINI-Phi-4"}

#output_folders = {"outputs_increase_new":"Increase_effect_final"}


traits = ['E', 'A', 'C', 'N', 'O']
fixed_traits = ['E','A','C','ES','I']

# features = ["model_name"]


# for bot_trait in traits:
#     for user_trait in traits:
#         features.append(f"{user_trait}-{bot_trait}_r")
#         features.append(f"{user_trait}-{bot_trait}_ci_bottom")
#         features.append(f"{user_trait}-{bot_trait}_ci_top")
#         features.append(f"{user_trait}-{bot_trait}_slope")

# all_data=[features]
fig = plt.figure(figsize=(36, 24))
gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.1], 
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.2, hspace=0.3)

axes = []
for i in range(2):
    for j in range(3):
        ax = fig.add_subplot(gs[i, j])
        axes.append(ax)

cbar_ax = fig.add_subplot(gs[:, -1])

# Store correlation matrices and confidence intervals for each model
model_data = {}
for folder, model_name in output_folders.items():
    data = []
    for j in range(1000):
        file_path = f"{folder}/output_{j}.txt"
        if os.path.exists(file_path):
            result = extract_data(file_path)
            if result:
                data.append(result)
    
    df = pd.DataFrame(data, columns=['bot_initial', 'bot_shift', 'user_initial', 'user_shift'])
    
    # Store correlations and CIs for this specific model
    correlations = {}
    corr_matrix = np.zeros((5, 5))
    ci_matrix = np.zeros((5, 5, 2))  # Store both lower and upper CI bounds
    
    for i, bot_trait in enumerate(traits):
        for j, user_trait in enumerate(traits):
            x = df['user_initial'].apply(lambda d: d[user_trait])
            y = df['bot_shift'].apply(lambda d: d[bot_trait])
            r, ci, slope = calculate_correlation(x, y)
            corr_matrix[i, j] = r
            ci_matrix[i, j] = ci  # Store CI for this specific correlation
    
    model_data[model_name] = {
        'correlations': corr_matrix,
        'confidence_intervals': ci_matrix
    }

# Create heatmaps
for idx, (model_name, data) in enumerate(model_data.items()):
    is_left = idx % 3 == 0
    is_bottom = idx >= 3
    
    corr_matrix = data['correlations']
    ci_matrix = data['confidence_intervals']
    
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=fixed_traits if is_bottom else False,
                yticklabels=fixed_traits if is_left else False,
                annot=False, ax=axes[idx], cbar=False)
    
    if is_bottom:
        axes[idx].tick_params(axis='x', labelsize=16, pad=10)
    if is_left:
        axes[idx].tick_params(axis='y', labelsize=16, pad=10)
    
    # Add correlation values with model-specific significance bold formatting
    for i in range(5):
        for j in range(5):
            r = corr_matrix[i, j]
            ci = ci_matrix[i, j]  # Get CI for this specific correlation in this model
            text = f"{r:.2f}"
            # Bold if CI doesn't include 0 (significant correlation)
            is_significant = (ci[0] > 0) or (ci[1] < 0)
            axes[idx].text(j + 0.5, i + 0.5, text,
                          horizontalalignment='center',
                          verticalalignment='center',
                          color='black',
                          fontsize=30,
                          fontweight='bold' if is_significant else 'normal')
    
    axes[idx].set_title(f"{model_name}", pad=30, fontsize=36)
    axes[idx].set_xlabel('')
    axes[idx].set_ylabel('')

# Create colorbar
norm = plt.Normalize(vmin=-1, vmax=1)
sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cbar_ax)
cbar.ax.tick_params(labelsize=16)
cbar.set_label('Correlation Coefficient', fontsize=20, labelpad=20)

# Add axis labels
fig.text(0.5, 0.02, 'Chatbot Trait Score Shift', ha='center', fontsize=24)
fig.text(0.06, 0.5, 'User Trait Score', va='center', rotation='vertical', fontsize=24)

# Set main title
plt.suptitle("User Trait vs Chatbot Trait Shift Linear Correlations Across Models",
             fontsize=44, y=0.98)

plt.savefig('combined_heatmaps_new.png', dpi=300, bbox_inches='tight')
plt.close()
print("Combined heatmap saved as 'combined_correlation_heatmaps.png'")
# # TO MAKE INDIVIDUAL/PER FOLDER ANALYSES

# #wildchat_folders = ["wildchat_outputs","updated_2_wildchat_outputs_LLM-A_gpt-4o-mini_LLM-B_gpt-4o-mini"]

# for folder,model_name in output_folders.items():
#     data = []
#     for j in range(500):
#         file_path = f"{folder}/output_{j}.txt"
#         if os.path.exists(file_path):
#             result = extract_data(file_path)
#             if result:
#                 data.append(result)

#     df = pd.DataFrame(data, columns=['bot_initial', 'bot_shift', 'user_initial', 'user_shift'])

#     correlations = {}
#     for bot_trait in traits:
#         for user_trait in traits:
#             x = df['user_initial'].apply(lambda d: d[user_trait])
#             y = df['bot_shift'].apply(lambda d: d[bot_trait])
#             r, ci, slope = calculate_correlation(x, y)
#             correlations[f"{user_trait}-{bot_trait}"] = (r, ci, slope)
    

#     # #MAKE CSV OF DATA
#     # row = [model_name]

#     # # user trait - chatbot shift
#     # for pair, (r, ci, slope) in correlations.items():
#     #     row.extend([r, ci[0], ci[1], slope])

#     # for trait in traits:
#     #     shifts = df['bot_shift'].apply(lambda d: d[trait])
#     #     avg_shift = np.mean(shifts)
#     #     std_shift = np.std(shifts)
#     #     all_data[0].append(f"bot_{trait}_shift_mean")
#     #     row.append(avg_shift)
#     #     all_data[0].append(f"bot_{trait}_shift_std")
#     #     row.append(std_shift)

#     # all_data.append(row)

#     # MAKE CORRELATION HEATMAPS

#     # Set global font sizes
#     plt.rcParams.update({
#         'font.size': 24,          # Increased base font size
#         'axes.titlesize': 30,     # Title font size
#         'axes.labelsize': 30,     # Axis label font size
#         'xtick.labelsize': 30,    # Increased X-axis tick labels
#         'ytick.labelsize': 30     # Increased Y-axis tick labels
#     })

#     corr_matrix = np.zeros((5, 5))
#     for i, bot_trait in enumerate(traits):
#         for j, user_trait in enumerate(traits):
#             corr_matrix[i, j] = correlations[f"{user_trait}-{bot_trait}"][0]

#     # Create larger figure
#     plt.figure(figsize=(16, 14))

#     # Create heatmap with larger font sizes
#     heatmap = sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1,
#                         xticklabels=fixed_traits, yticklabels=fixed_traits, annot=False)

#     # Increase colorbar tick labels
#     heatmap.figure.axes[-1].tick_params(labelsize=22)

#     # Add correlation values with larger font
#     for i in range(5):
#         for j in range(5):
#             r = corr_matrix[i, j]
#             ci = correlations[f"{traits[j]}-{traits[i]}"][1]
            
#             text = f"{r:.2f}"
            
#             plt.text(j + 0.5, i + 0.5, text,
#                     horizontalalignment='center',
#                     verticalalignment='center',
#                     color='black',
#                     fontsize=30, fontweight='bold' if (ci[0]>0 or ci[1]<0) else 'normal')  # Significantly increased font size for correlation values

#     # Rotate x-axis labels for better spacing if needed
#     plt.xticks(rotation=45, ha='right')

#     plt.title(f"Increase Mirroring Correlations: User Trait vs Chatbot Trait Shift", pad=25)
#     plt.xlabel("User Initial Trait", labelpad=15)
#     plt.ylabel("Chatbot Trait Shift", labelpad=15)

#     # Adjust layout to prevent text cutoff
#     plt.tight_layout()

#     # Save with higher DPI and larger size
#     plt.savefig(f'combined_wildchat_correlation_ci_heatmap.png', 
#                 dpi=300, 
#                 bbox_inches='tight',
#                 pad_inches=0.5)
#     plt.close()
#     # corr_matrix = np.zeros((5, 5))
#     # for i, bot_trait in enumerate(traits):
#     #     for j, user_trait in enumerate(traits):
#     #         corr_matrix[i, j] = correlations[f"{user_trait}-{bot_trait}"][0]

#     # plt.figure(figsize=(12, 10))

#     # sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1,
#     #             xticklabels=traits, yticklabels=traits, annot=False)

#     # for i in range(5):
#     #     for j in range(5):
#     #         r = corr_matrix[i, j]
#     #         ci = correlations[f"{traits[j]}-{traits[i]}"][1]
            
#     #         text = f"{r:.2f}"
            
#     #         plt.text(j + 0.5, i + 0.5, text,
#     #                 horizontalalignment='center',
#     #                 verticalalignment='center',
#     #                 color='black' if abs(r) < 0.5 else 'white')

#     # plt.title(f"{model_name} Correlation Heatmap: User Initial vs Bot Shift")
#     # plt.xlabel("User Initial Trait")
#     # plt.ylabel("Bot Shift Trait")

#     # plt.tight_layout()
#     # plt.savefig(f'{model_name}_correlation_ci_heatmap.png', dpi=300, bbox_inches='tight')
#     # plt.close()

#     #print(f"Heatmap saved as '{model_name}_correlation_ci_heatmap.png'")

#     # # MAKE SCATTERPLOTS
#     # fig, axes = plt.subplots(5, 5, figsize=(20, 20))
#     # fig.suptitle(f"{model_name} Raw Data Scatter Plots: User Initial vs Bot Shift", y=1.02, fontsize=16)

#     # for i, bot_trait in enumerate(traits):
#     #     for j, user_trait in enumerate(traits):
#     #         ax = axes[i, j]
            
#     #         x = df['user_initial'].apply(lambda d: d[user_trait])
#     #         y = df['bot_shift'].apply(lambda d: d[bot_trait])

#     #         ax.scatter(x, y, alpha=0.1, s=20)

#     #         r = correlations[f"{user_trait}-{bot_trait}"][0]
#     #         ax.text(0.05, 0.95, f'r = {r:.3f}', 
#     #                 transform=ax.transAxes, 
#     #                 verticalalignment='top')
            
#     #         if i == 4:
#     #             ax.set_xlabel(f'User Initial {user_trait}')
#     #         if j == 0:
#     #             ax.set_ylabel(f'Bot Shift {bot_trait}')
                
#     #         ax.grid(True, alpha=0.3)
            
#     #         ax.set_aspect('equal', adjustable='box')

#     # plt.tight_layout()
#     # plt.savefig(f'{model_name}_scatter_plots.png', dpi=300, bbox_inches='tight')
#     # plt.close()

#     # print(f"Scatter plots saved as '{model_name}_scatter_plots.png'")

# # # MAKE CSV
# # with open(f"base_analysis_results.csv", "w", newline="") as file:
# #     writer = csv.writer(file)
# #     writer.writerows(all_data)

