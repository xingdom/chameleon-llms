import tiktoken, re, os
from transformers import AutoTokenizer
from functools import lru_cache
from tqdm import tqdm


# Maps model name used in directory path to models offical repository name
model_nickname_to_fullname = {
    "gemma": "google/gemma-2-2b-it",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "llama":"meta-llama_Meta-Llama-3.1-70B-Instruct"
}

@lru_cache(maxsize=30)
def get_tokenizer(model_fullname): 
    """Get the tokenizer. GPT-family models use the tiktoken library, other local models use AutoTokenizer from transformers."""
    if "gpt" in model_fullname.lower(): 
        tokenizer = tiktoken.encoding_for_model(model_fullname)
    else: 
        tokenizer = AutoTokenizer.from_pretrained(model_fullname)
    return tokenizer


def get_num_tokens(text, tokenizer): 
    """Uses tokenizer to get number of tokens in text."""
    ids = tokenizer.encode(text)
    return len(ids)


def parse_conversation(text):
    """
    Parse a conversation transcript into a list of sentences, 
    one per turn. Assumes turns are labeled like '(Turn 0)', '(Turn 1)', etc.
    """
    # Regex captures turn index and content after the colon
    pattern = re.compile(r"\(Turn\s+\d+\)\s*(?:LLM [AB]:\s*)?(.*)")
    
    sentences = []
    for match in pattern.finditer(text):
        # Strip quotes and whitespace
        sentence = match.group(1).strip()
        if sentence.startswith('"') and sentence.endswith('"'):
            sentence = sentence[1:-1].strip()
        if sentence.startswith("LLM A: ") or sentence.startswith("LLM B: "):
            sentence = sentence[7:]
        sentences.append(sentence)
    
    return sentences


# Example usage:
if __name__ == "__main__":

    MODEL_A = "gemma"
    MODEL_B = "gpt-4o-mini"

    directory_name = f"outputs_LLM-A_{MODEL_A}_LLM-B_{MODEL_B}"

    length_counter_dict = {i:0 for i in range(101)}

    for run_idx in tqdm(range(1000)):
        convo_filepath = f"{directory_name}/output_{run_idx}.txt"

        with open(convo_filepath, "r", encoding="utf-8") as f:
            data = f.read()

        turns = parse_conversation(data)
        for i, turn in enumerate(turns):
            if i % 2 == 0: 
                model_fullname = model_nickname_to_fullname[MODEL_B]
            else: 
                model_fullname = model_nickname_to_fullname[MODEL_A]

            tok = get_tokenizer(model_fullname)
            num_ids = get_num_tokens(turn, tok)

            length_counter_dict[num_ids] += 1

print(length_counter_dict)
print(f"Number of turns where 100 output tokens are generated: {length_counter_dict[100]}")