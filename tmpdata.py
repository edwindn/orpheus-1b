import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from huggingface_hub import login as hf_login, snapshot_download
import wandb
import dotenv

"""
run on cpu
"""

dotenv.load_dotenv()

hf_login(os.getenv("HF_TOKEN_EDWIN"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MAX_SEQ_LENGTH = 8192
CPU_COUNT = os.cpu_count()
TRAIN_BATCH_SIZE = 1

# ---------------------- #

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
tokenizer.pad_token = tokenizer.eos_token  # Set pad token to EOS token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
model = model.to(device)  # Move model to device

llama_token_end = 128256
snac_vocab_size = 4096
start_of_text = 128000
end_of_text = 128001

start_of_human = llama_token_end + 1
end_of_human = llama_token_end + 2

start_of_gpt = llama_token_end + 3
end_of_gpt = llama_token_end + 4

start_of_audio = llama_token_end + 5
end_of_audio = llama_token_end + 6

pad_token = llama_token_end + 7

audio_token_start = llama_token_end + 10

new_vocab_size = audio_token_start + 7 * snac_vocab_size

# ---------------------- #

# Resize token embeddings
model.resize_token_embeddings(new_vocab_size)
model.config.vocab_size = new_vocab_size

# Download dataset
dataset_path = snapshot_download(
    repo_id="edwindn/emilia-snac-orpheus-1b-unpadded",
    repo_type="dataset",
    revision="main",
    max_workers=CPU_COUNT,
)

dataset = load_dataset(dataset_path, split="train")

def preprocess_map(example):
    tokens = example['tokens']
    return {
        'input_ids': tokens,
        'attention_mask': [1] * len(tokens),
        'labels': tokens.copy()
    }


dataset = dataset.map(preprocess_map, batched=False, num_proc=CPU_COUNT, remove_columns=["tokens"])

train_dataset = dataset.shuffle(seed=42)

hf_login(os.getenv("HF_TOKEN_EDWIN"))

train_dataset.push_to_hub(
    "edwindn/emilia-snac-orpheus-1b-unpadded",
    split="train",
    private=True
)
