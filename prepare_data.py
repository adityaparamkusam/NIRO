# prepare_data.py
# This script reads raw .txt files, trains a SentencePiece tokenizer,
# and then tokenizes all text, saving the token IDs into .pt binary files
# for efficient loading during model training.

import os
import glob
import sentencepiece as spm
import torch
from tqdm import tqdm

# --- Configuration Parameters ---
DATA_FOLDER = "/Users/adityamadhava/Downloads/Techno/NIRO/ENT_DATA"  # Input folder containing your raw .txt files
TOKENIZED_DATA_FOLDER = "tokenized_data"  # Output folder for tokenized .pt files
SPM_MODEL_PREFIX = "niro_tokenizer"  # Prefix for the SentencePiece model files
VOCAB_SIZE = 16000 # Vocabulary size for SentencePiece tokenizer
TOKENIZER_MODEL_PATH = f"{SPM_MODEL_PREFIX}.model"


# --- Functions for Data Preparation ---

def get_text_files_from_folder(folder_path):
    """Collects all .txt file paths from the specified folder."""
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Error: Data folder '{folder_path}' not found. Please create it and place your .txt files inside.")
    file_paths = glob.glob(os.path.join(folder_path, '*.txt'))
    if not file_paths:
        raise ValueError(f"Error: No .txt files found in '{folder_path}'. Please ensure your data is in .txt format within this folder.")
    print(f"Found {len(file_paths)} .txt files in '{folder_path}'.")
    return file_paths


def train_sentencepiece_tokenizer(text_file_paths, model_prefix, vocab_size):
    """
    Trains a SentencePiece tokenizer model on the given text files.
    """
    print(f"Training SentencePiece tokenizer with vocab size {vocab_size} on {len(text_file_paths)} files...")
    # SentencePiece command line options; input takes a comma-separated list of files
    cmd = (
        f"--input={','.join(text_file_paths)} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "  # Using Byte Pair Encoding (BPE)
        f"--character_coverage=1.0 "  # Ensures all characters are covered
        f"--num_threads={os.cpu_count()} "  # Use all available CPU cores for training
        f"--normalization_rule_name=nmt_nfkc_cf "  # Recommended normalization rule for NMT
    )
    spm.SentencePieceTrainer.train(cmd)
    print(f"SentencePiece tokenizer training complete. Model saved to {model_prefix}.model")

    # Load the trained model to verify
    spm_processor = spm.SentencePieceProcessor()
    spm_processor.load(f"{model_prefix}.model")
    return spm_processor


def tokenize_and_save_files(text_file_paths, tokenizer, output_folder):
    """
    Tokenizes each text file and saves the resulting token IDs as individual .pt files.
    """
    os.makedirs(output_folder, exist_ok=True)  # Create output directory if it doesn't exist
    print(f"Tokenizing {len(text_file_paths)} files and saving to '{output_folder}'...")

    for file_path in tqdm(text_file_paths, desc="Tokenizing & Saving"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Tokenize the text into a list of integer IDs
            ids = tokenizer.encode_as_ids(text)

            # Convert to PyTorch tensor
            token_tensor = torch.tensor(ids, dtype=torch.long)

            # Construct output file path
            # Use original filename (without .txt extension) + .pt
            base_name = os.path.basename(file_path)
            file_name_without_ext = os.path.splitext(base_name)[0]
            output_file_path = os.path.join(output_folder, f"{file_name_without_ext}.pt")

            # Save the tensor
            torch.save(token_tensor, output_file_path)

        except Exception as e:
            print(f"Error processing and saving {file_path}: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    print("--- Starting Data Preparation for NIRO Project ---")

    # 1. Collect all .txt files from the data folder
    input_text_files = get_text_files_from_folder(DATA_FOLDER)

    # 2. Train the SentencePiece tokenizer if not already trained
    if not os.path.exists(TOKENIZER_MODEL_PATH):
        tokenizer_processor = train_sentencepiece_tokenizer(
            input_text_files, SPM_MODEL_PREFIX, VOCAB_SIZE
        )
    else:
        print(f"Tokenizer model already exists at {TOKENIZER_MODEL_PATH}. Loading existing model.")
        tokenizer_processor = spm.SentencePieceProcessor()
        tokenizer_processor.load(TOKENIZER_MODEL_PATH)

    # 3. Tokenize all text files and save them as .pt files
    tokenize_and_save_files(input_text_files, tokenizer_processor, TOKENIZED_DATA_FOLDER)

    print("--- Data Preparation Complete! ---")
    print(f"Tokenized data saved to '{TOKENIZED_DATA_FOLDER}/'")
    print(f"Tokenizer model saved to '{TOKENIZER_MODEL_PATH}'")
