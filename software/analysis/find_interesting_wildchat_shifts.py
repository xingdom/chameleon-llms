import os
import re
import ast

def extract_final_dict_from_brackets(file_path):
    """Extracts the final matched curly bracket content and converts it into a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regex pattern to match non-nested curly brackets
    pattern = r"\{.*?\}"

    matches = re.findall(pattern, content)  # Find all non-nested bracket pairs

    if not matches:
        print(f"No curly brackets found in {file_path}")
        return None

    final_match = matches[-1]  # Get the last matched bracket content
    final_content = final_match[1:-1].strip()  # Remove outer { }

    try:
        dictionary_obj = ast.literal_eval("{" + final_content + "}")  # Convert to dictionary
        if isinstance(dictionary_obj, dict):
            return dictionary_obj
        else:
            print(f"Invalid dictionary format in {file_path}")
            return None
    except (SyntaxError, ValueError):
        print(f"Could not parse dictionary in {file_path}")
        return None


def process_directory(directory_path):
    """Loops through .txt files in the directory and processes each one."""
    results = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory_path, filename)
            extracted_dict = extract_final_dict_from_brackets(file_path)
            
            if extracted_dict is not None:
                for trait,value in extracted_dict.items():
                    if (value >= 10 or value <= -10) and trait != 'A':
                        print(filename)
                        break
    # return results


# Example Usage
directory_path = "Results/wildchat_outputs_LLM-A_gpt-4o-mini_LLM-B_gpt-4o-mini"  # Replace with your directory path
process_directory(directory_path)
