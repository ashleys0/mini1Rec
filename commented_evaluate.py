"""
Constrained Language Model Recommendation System

This script uses a language model (like LLaMA or GPT-2) to generate product recommendations
with constrained decoding. The constraint ensures the model only generates valid item IDs
or titles from a predefined catalog, preventing hallucination of non-existent items.

Key Concepts:
- Prefix Tree (Trie): A hash-based structure that stores all valid token sequences
- Constrained Generation: Forces the model to only output tokens that lead to valid items
- Beam Search: Explores multiple candidate sequences simultaneously
"""

import pandas as pd
import fire
import torch
import json
import os
from transformers import GenerationConfig,  AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, LogitsProcessorList, TemperatureLogitsWarper
from data import  EvalD3Dataset, EvalSidDataset
from LogitProcessor import ConstrainedLogitsProcessor
from accelerate import Accelerator
import random
import bitsandbytes as bnb


# Determine if GPU is available for faster inference
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Constants for hashing (likely not used, but defined for potential hash functions)
P = 998244353  # Large prime number for hashing
MOD = int(1e9 + 9)  # Modulo value for hash overflow prevention
import numpy as np


def get_hash(x):
    """
    Convert a sequence of token IDs into a hash string key.
    
    This creates a unique string identifier for each token sequence, which is used
    as a key in the prefix tree dictionary. The hash allows fast O(1) lookup to find
    which tokens are valid at each position.
    
    Args:
        x: List of token IDs (integers)
    
    Returns:
        String hash in format "id1-id2-id3-..." (e.g., "123-456-789")
    
    Example:
        get_hash([1, 2, 3]) -> "1-2-3"
    """
    x = [str(_) for _ in x]  # Convert each token ID to string
    return '-'.join(x)  # Join with hyphens to create unique key


def set_seed(seed):
    """
    Set random seeds for reproducibility across all libraries.
    
    Machine learning models use random number generators for initialization,
    sampling, etc. Setting seeds ensures experiments can be reproduced exactly.
    
    Args:
        seed: Integer seed value (e.g., 42)
    """
    random.seed(seed)  # Python's built-in random module
    np.random.seed(seed)  # NumPy's random number generator
    torch.manual_seed(seed)  # PyTorch CPU random number generator
    
    if torch.cuda.is_available():
        # Set seeds for all CUDA devices (GPUs)
        torch.cuda.manual_seed(seed)  # Current GPU
        torch.cuda.manual_seed_all(seed)  # All GPUs if using multi-GPU
    
    # Optional: Make cudnn deterministic (commented out for performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    
def main(
    base_model: str = "",  # Path to pretrained model (e.g., "meta-llama/Llama-2-7b")
    train_file: str = "",  # Not used in this script (legacy parameter)
    info_file: str = "",  # Path to item catalog file (format: semantic_id \t title \t item_id)
    category: str = "",  # Product category key (e.g., "Books", "Toys_and_Games")
    test_data_path: str = "",  # Path to test dataset with user histories
    result_json_data: str = "",  # Output path for predictions JSON file
    batch_size: int = 4,  # Number of users to process simultaneously
    K: int = 0,  # Number of items in user's interaction history to use
    seed: int = 42,  # Random seed for reproducibility
    length_penalty: float = 0.0,  # Beam search parameter: <1 prefers shorter, >1 prefers longer
    max_new_tokens: int = 256,  # Maximum tokens to generate per recommendation
    num_beams: int = 50,  # Number of parallel beam search hypotheses
):
    """
    Main function to run constrained recommendation generation.
    
    This function:
    1. Loads a pretrained language model
    2. Builds a prefix tree of all valid item IDs and titles
    3. Runs constrained beam search to generate recommendations
    4. Ensures all generated recommendations are valid items from the catalog
    
    The key innovation is using constrained decoding: at each generation step,
    the model can only choose tokens that eventually lead to a valid item.
    This prevents hallucination of non-existent items.
    
    Args:
        base_model: HuggingFace model path or name
        info_file: TSV file with item catalog (semantic_id \t title \t item_id per line)
        category: Category name that gets mapped to human-readable text
        test_data_path: JSON/pickle file with user interaction histories
        result_json_data: Where to save the recommendation results
        batch_size: How many users to process in parallel (limited by GPU memory)
        K: How many past user interactions to include as context
        seed: Random seed for reproducibility
        length_penalty: Beam search length penalty (1.0 = neutral)
        max_new_tokens: Max length of generated recommendation text
        num_beams: Number of candidate sequences to maintain in beam search
    """
    # Set all random seeds for reproducibility
    random.seed(seed)
    set_seed(seed)
    
    # Map internal category codes to human-readable descriptions
    # These descriptions are likely used in prompts to the language model
    category_dict = {
        "Industrial_and_Scientific": "industrial and scientific items", 
        "Office_Products": "office products", 
        "Toys_and_Games": "toys and games", 
        "Sports": "sports and outdoors", 
        "Books": "books"
    }
    category = category_dict[category]  # Convert code to readable name
    print(category)

    # Load the pretrained language model
    # torch_dtype=torch.bfloat16: Use 16-bit precision to save memory
    # device_map="auto": Automatically distribute model across available GPUs
    model = AutoModelForCausalLM.from_pretrained(
        base_model, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    model.eval()  # Set to evaluation mode (disables dropout, etc.)
    
    # Load and parse the item catalog file
    with open(info_file, 'r') as f:
        info = f.readlines()
        
        # Parse the new format: semantic_id \t item_title \t item_id
        # semantic_ids: List of item semantic IDs (e.g., ["1-2-3\n", "4-5-6\n"])
        semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
        
        # item_titles: List of human-readable item titles
        item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]
        
        # Format items as model responses for tokenization
        # This wraps each item in the expected output format
        info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
        info_titles = [f'''### Response:\n{_}''' for _ in item_titles]

    # Load the tokenizer (converts text to token IDs and vice versa)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Tokenize all semantic IDs to build the prefix tree
    if base_model.lower().find("llama") > -1:
        # LLaMA models prepend a special token, so skip it with [1:]
        prefixID = [tokenizer(_).input_ids[1:] for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids[1:] for _ in info_titles]
    else:
        # Other models (like GPT-2) don't need special handling
        prefixID = [tokenizer(_).input_ids for _ in info_semantic]
        prefixTitleID = [tokenizer(_).input_ids for _ in info_titles]
    
    # Determine where the actual item ID starts in the token sequence
    # This skips the prompt tokens like "### Response:\n"
    if base_model.lower().find("gpt2") > -1:
        prefix_index = 4  # GPT-2 uses 4 tokens for the prefix
    else:
        prefix_index = 3  # Most other models use 3 tokens
    
    # Build hash_dict: A prefix tree (trie) for semantic IDs
    # This dictionary maps partial token sequences to valid next tokens
    # Key: hash of token sequence, Value: list of valid next tokens
    hash_dict = dict()
    
    for index, ID in enumerate(prefixID):
        # Append EOS (end-of-sequence) token to mark end of item ID
        ID.append(tokenizer.eos_token_id)
        
        # Build the prefix tree by adding all prefixes of this ID
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                # First token: hash includes the prompt prefix
                hash_number = get_hash(ID[:i])
            else:
                # Subsequent tokens: hash only includes the item ID portion
                hash_number = get_hash(ID[prefix_index:i])
            
            # Add the next valid token to the set for this prefix
            if hash_number not in hash_dict:
                hash_dict[hash_number] = set()  # Use set to avoid duplicates
            hash_dict[hash_number].add(ID[i])  # ID[i] is the next valid token
        
        # Also store the complete sequence hash
        hash_number = get_hash(ID[prefix_index:])

    # Build hash_dict_title: A separate prefix tree for item titles
    # This allows the model to generate either IDs or full text titles
    hash_dict_title = dict()
    for index, ID in enumerate(prefixTitleID):
        ID.append(tokenizer.eos_token_id)
        
        # Same logic as above but for title tokens
        for i in range(prefix_index, len(ID)):
            if i == prefix_index:
                hash_number = get_hash(ID[:i])
            else:
                hash_number = get_hash(ID[prefix_index:i])
            if hash_number not in hash_dict_title:
                hash_dict_title[hash_number] = set()
            hash_dict_title[hash_number].add(ID[i])
        hash_number = get_hash(ID[prefix_index:])

    # Convert sets to lists for compatibility with the generation library
    for key in hash_dict.keys():
        hash_dict[key] = list(hash_dict[key])
    for key in hash_dict_title.keys():
        hash_dict_title[key] = list(hash_dict_title[key])

    # Define constraint functions for constrained generation
    # These functions are called during beam search to limit valid next tokens
    
    def prefix_allowed_tokens_fn_semantic(batch_id, input_ids):
        """
        Constraint function for semantic ID generation.
        
        At each generation step, this function is called to determine which
        tokens are valid. It looks up the current sequence in the prefix tree
        and returns only tokens that lead to valid semantic IDs.
        
        Args:
            batch_id: Index of the current sequence in the batch (unused here)
            input_ids: Current sequence of generated token IDs
        
        Returns:
            List of valid next token IDs, or empty list if no valid continuations
        """
        hash_number = get_hash(input_ids)  # Convert current sequence to hash key
        if hash_number in hash_dict:
            return hash_dict[hash_number]  # Return valid next tokens
        return []  # No valid continuations (dead end)
        
    def prefix_allowed_tokens_fn_title(batch_id, input_ids):
        """
        Constraint function for item title generation.
        
        Same as semantic function but uses the title prefix tree instead.
        This allows generating human-readable item titles rather than IDs.
        
        Args:
            batch_id: Index of current sequence in batch
            input_ids: Current sequence of generated token IDs
        
        Returns:
            List of valid next token IDs for title generation
        """
        hash_number = get_hash(input_ids)
        if hash_number in hash_dict_title:
            return hash_dict_title[hash_number]
        return []

    # Set which constraint function to use (semantic IDs by default)
    # Uncomment the second line to use title generation instead
    prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_semantic
    # prefix_allowed_tokens_fn = prefix_allowed_tokens_fn_title
    
    # Configure tokenizer padding behavior
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"  # Pad on left so recent tokens are at the end
    
    # Load the evaluation dataset
    # This contains user interaction histories and the items to recommend
    val_dataset = EvalSidDataset(
        train_file=test_data_path,  # Path to test data
        tokenizer=tokenizer,  # Tokenizer for text processing
        max_len=2560,  # Maximum sequence length
        category=category,  # Product category
        test=True,  # Flag indicating this is test (not training) data
        K=K,  # Number of historical items to include
        seed=seed  # Random seed
    )
    
    # Get all tokenized inputs from the dataset
    encodings = [val_dataset[i] for i in range(len(val_dataset))]
    
    # Get the raw test data (with ground truth for evaluation)
    test_data = val_dataset.get_all()

    # Configure model tokens to match tokenizer
    model.config.pad_token_id = model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    def evaluate(
            encodings,  # List of tokenized inputs (one per user)
            num_beams=10,  # Number of beam search hypotheses
            max_new_tokens=64,  # Maximum tokens to generate
            length_penalty=1.0,  # Beam search length penalty
            **kwargs,  # Additional generation parameters
    ):
        """
        Run constrained generation to produce recommendations.
        
        This function:
        1. Batches and pads the inputs to the same length
        2. Configures beam search with constraints
        3. Generates recommendations using the language model
        4. Decodes token IDs back to text
        
        The key component is the ConstrainedLogitsProcessor, which modifies
        the model's output probabilities at each step to only allow tokens
        that lead to valid items.
        
        Args:
            encodings: List of dictionaries with "input_ids" keys
            num_beams: Number of parallel sequences in beam search
            max_new_tokens: Maximum length of generated sequence
            length_penalty: Penalty for longer sequences (1.0 = neutral)
            **kwargs: Additional arguments passed to GenerationConfig
        
        Returns:
            List of lists, where each inner list contains num_beams recommendations
            for one user (e.g., [[rec1, rec2, ...], [rec1, rec2, ...], ...])
        """
        # Find the longest input sequence for padding
        maxLen = max([len(_["input_ids"]) for _ in encodings])

        # Prepare padded inputs for batched processing
        padding_encodings = {"input_ids": []}
        attention_mask = []  # Mask to ignore padding tokens

        for _ in encodings:
            L = len(_["input_ids"])  # Current sequence length
            
            # Pad on the left with pad tokens to reach maxLen
            padding_encodings["input_ids"].append(
                [tokenizer.pad_token_id] * (maxLen - L) + _["input_ids"]
            )
            
            # Create attention mask: 0 for padding, 1 for real tokens
            attention_mask.append([0] * (maxLen - L) + [1] * L)
        
        # Configure the generation strategy
        generation_config = GenerationConfig(
            num_beams=num_beams,  # How many hypotheses to track
            length_penalty=length_penalty,  # Preference for longer/shorter sequences
            num_return_sequences=num_beams,  # Return all beam search results
            pad_token_id=model.config.pad_token_id,
            eos_token_id=model.config.eos_token_id,
            max_new_tokens=max_new_tokens,  # Maximum length to generate
            **kwargs  # Any additional parameters
        )
        
        # Run generation without computing gradients (inference only)
        with torch.no_grad():
            # Create the constrained logits processor
            # This modifies the model's output probabilities to enforce constraints
            clp = ConstrainedLogitsProcessor(
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,  # Constraint function
                num_beams=num_beams,  # Must match generation config
                base_model=base_model  # Model name for any model-specific logic
            )
            logits_processor = LogitsProcessorList([clp])  # Wrap in list

            # Run constrained beam search generation
            generation_output = model.generate(
                torch.tensor(padding_encodings["input_ids"]).to(device),  # Input token IDs
                attention_mask=torch.tensor(attention_mask).to(device),  # Attention mask
                generation_config=generation_config,  # Generation settings
                return_dict_in_generate=True,  # Return detailed output
                output_scores=True,  # Include token scores
                logits_processor=logits_processor,  # Apply constraints
            )
       
        # Extract just the newly generated tokens (remove the input prompt)
        batched_completions = generation_output.sequences[:, maxLen:]
       
        # Decode token IDs back to text
        output = tokenizer.batch_decode(
            batched_completions, 
            skip_special_tokens=True
        )
        
        # Clean up the output text
        # Remove the "Response:\n" prefix that was part of the format
        output = [_.split("Response:\n")[-1].strip() for _ in output]
        
        # Group outputs by user (each user has num_beams recommendations)
        # Shape: (num_users, num_beams)
        real_outputs = [
            output[i * num_beams: (i + 1) * num_beams] 
            for i in range(len(output) // num_beams)
        ]
        return real_outputs
    
    # Move model to GPU 
    model = model.to(device)

    # Run evaluation on all test users
    from tqdm import tqdm
    outputs = []  # Store all recommendations
    new_encodings = []  # Store batched inputs
    
    # Split encodings into batches
    BLOCK = (len(encodings) + batch_size - 1) // batch_size  # Ceiling division
    for i in range(BLOCK):
        # Create batches of size batch_size (last batch may be smaller)
        new_encodings.append(encodings[i * batch_size: (i + 1) * batch_size])

    # Process each batch with progress bar
    for idx, encodings in enumerate(tqdm(new_encodings)):
        # Generate recommendations for this batch
        output = evaluate(
            encodings, 
            max_new_tokens=max_new_tokens, 
            num_beams=num_beams, 
            length_penalty=length_penalty
        )
        
        # Accumulate all outputs
        outputs = outputs + output
    
    # Add predictions to the test data
    for i, test in enumerate(test_data):
        test["predict"] = outputs[i]  # Store generated recommendations
  
    # Clean up test data by removing internal 'dedup' field if present
    for i in range(len(test_data)):
        if 'dedup' in test_data[i]:
            test_data[i].pop('dedup')
    
    # Save results to JSON file
    with open(result_json_data, 'w') as f:
        json.dump(test_data, f, indent=4)  # Pretty print with 4-space indentation


# Entry point when script is run from command line
if __name__ == '__main__':
    # Use Python Fire to automatically create CLI from main() parameters
    # Example: python script.py --base_model=meta-llama/Llama-2-7b --num_beams=50
    fire.Fire(main)