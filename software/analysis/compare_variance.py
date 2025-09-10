import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple

def extract_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
   
    if len(lines) < 4:
        return None
   
    try:
        match = re.search(r'\{.*\}', lines[-3])
        if match:
            d = eval(match.group())
            if all(key in d for key in ['E', 'A', 'C', 'N', 'O']):
                return d
            else:
                return None
        else:
            return None
    except:
        return None

def collect_data_from_folders(folders: List[str]) -> List[Dict]:
    """Collect data from multiple folders."""
    data = []
    for folder in folders:
        for j in range(1000):  # Adjust range based on expected files
            file_path = f"{folder}/output_{j}.txt"
            if os.path.exists(file_path):
                result = extract_data(file_path)
                if result:
                    data.append(result)
    return data

def calculate_statistics(data: List[Dict], group_name: str) -> pd.DataFrame:
    """Calculate descriptive statistics for a dataset."""
    df = pd.DataFrame(data)
    stats_dict = {
        'group': group_name,
        'count': len(df),
        'mean': df.mean(),
        'std': df.std(),
        'variance': df.var()
    }
    return pd.DataFrame(stats_dict)

def compare_variances(df1: pd.DataFrame, df2: pd.DataFrame, traits: List[str]) -> pd.DataFrame:
    """Perform statistical tests comparing variances between two datasets."""
    results = []
    
    for trait in traits:
        # Levene's test (more robust than F-test)
        levene_stat, levene_p = stats.levene(df1[trait], df2[trait])
        
        # Calculate relative difference in standard deviations
        std_diff_percent = ((df2[trait].std() - df1[trait].std()) / df1[trait].std()) * 100
        
        results.append({
            'trait': trait,
            'wildchat_std': df1[trait].std(),
            'comparison_std': df2[trait].std(),
            'std_diff_percent': std_diff_percent,
            'levene_statistic': levene_stat,
            'levene_p_value': levene_p,
            'significantly_different': levene_p < 0.05
        })
    
    return pd.DataFrame(results)

# Define paths and traits
traits = ['E', 'A', 'C', 'N', 'O']
wildchat_folders = ["wildchat_outputs", "updated_2_wildchat_outputs_LLM-A_gpt-4o-mini_LLM-B_gpt-4o-mini"]
output_folders = ["outputs_LLM-A_gpt-4o_LLM-B_gpt-4o-mini","outputs_LLM-A_gpt-4o-mini_LLM-B_gpt-4o-mini","outputs_LLM-A_meta-llama_Meta-Llama-3.1-8B-Instruct_LLM-B_gpt-4o-mini","outputs_LLM-A_microsoft_phi-4_LLM-B_gpt-4o-mini","outputs_LLM-A_Qwen_Qwen2.5-7B-Instruct_LLM-B_gpt-4o-mini","outputs_LLM-A_mistralai_Mistral-Small-24B-Instruct-2501_LLM-B_gpt-4o-mini","outputs_LLM-A_google_gemma-2-2b-it_LLM-B_gpt-4o-mini"]

# Collect data from both experiments
print("Collecting data...")
wildchat_data = collect_data_from_folders(wildchat_folders)
comparison_data = collect_data_from_folders(output_folders)

# Convert to DataFrames
wildchat_df = pd.DataFrame(wildchat_data)
comparison_df = pd.DataFrame(comparison_data)

# Calculate basic statistics
print("\nBasic Statistics:")
print("-----------------")
wildchat_stats = calculate_statistics(wildchat_data, "Wildchat")
comparison_stats = calculate_statistics(comparison_data, "Comparison")

# Perform variance comparison
print("\nVariance Comparison Results:")
print("---------------------------")
variance_comparison = compare_variances(wildchat_df, comparison_df, traits)

# Format and display results
pd.set_option('display.float_format', '{:.3f}'.format)
print("\nSample Sizes:")
print(f"Wildchat dataset: {len(wildchat_data)} files")
print(f"Comparison dataset: {len(comparison_data)} files")

print("\nDetailed Comparison by Trait:")
print(variance_comparison[['trait', 'wildchat_std', 'comparison_std', 
                         'std_diff_percent', 'levene_p_value', 
                         'significantly_different']].to_string(index=False))

# Optional: Save results to CSV
variance_comparison.to_csv('variance_comparison_results.csv', index=False)