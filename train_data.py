from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download, login as hf_login
import os
import dotenv
from tqdm import tqdm
import random
import multiprocessing as mp
from functools import partial

dotenv.load_dotenv()

"""
run on cpu

possible changes:
add up to 2* len_snac for lower levels of tokens
"""

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MAX_SEQ_LENGTH = 8192
CPU_COUNT = os.cpu_count()
TRAIN_BATCH_SIZE = 1
PAD_TO_LENGTH = True
NUM_CHUNKS = 5

hf_login(os.getenv("HF_TOKEN_AMUVARMA"))
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

# ---------------------- #

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

# ---------------------- #

#ylacombe/emilia-subset
#repo_id = "amuvarma/emilia-snac-merged-all-TTS-grouped-8192"
repo_id = "amuvarma/em-EN"
#amuvarma/text-messages-6m-processed-1

snapshot_download(
    repo_id=repo_id,
    repo_type="dataset",
    revision="main",        
    max_workers=CPU_COUNT,
) 

dataset = load_dataset(repo_id, split="train")

def tokenize_map(entry):
    audio_tokens = entry['codes_list']
    text = entry['text']
    text_tokens = tokenizer(text).input_ids

    start = [start_of_human] if text_tokens[0] == start_of_text else [start_of_human, start_of_text]
    middle = [end_of_text, end_of_human, start_of_gpt, start_of_audio]
    end = [end_of_audio, end_of_gpt]
    tokens = start + text_tokens + middle + audio_tokens + end
    
    return {
        'input_ids': tokens,
        'attention_mask': [1] * len(tokens),
        'labels': tokens.copy()
    }


dataset = dataset.map(tokenize_map, batched=False, remove_columns=dataset.column_names, num_proc=CPU_COUNT)

if not PAD_TO_LENGTH:
    dataset = dataset.shuffle(seed=42)
    hf_login(os.getenv("HF_TOKEN_EDWIN"))

    dataset.push_to_hub(
        "edwindn/emilia-snac-orpheus-1b-unpadded",
        split="train",
        private=True
    )

    quit()


def process_chunk(dataset_chunk, dcix):
    train_dataset_chunk = []
    current_len = 0
    last_chunk = []

    for row in tqdm(dataset_chunk, desc=f"Processing chunk {dcix}"):
        tokens = row['input_ids']
            
        while current_len + len(tokens) > MAX_SEQ_LENGTH:
            needed = MAX_SEQ_LENGTH - current_len
            last_chunk.extend(tokens[:needed])
            train_dataset_chunk.append(
                {
                    'input_ids': last_chunk,
                    'attention_mask': [1] * len(last_chunk),
                    'labels': last_chunk.copy()
                }
            )

            if random.random() < 0.0001:
                print(f"Chunk {len(train_dataset_chunk)} for dataset chunk {dcix}: {len(last_chunk)}")
            
            tokens = tokens[needed:]
            last_chunk = []
            current_len = 0
        
        last_chunk.extend(tokens)
        current_len += len(tokens)
        
        if current_len == MAX_SEQ_LENGTH:
            train_dataset_chunk.append(
                {
                    'input_ids': last_chunk,
                    'attention_mask': [1] * len(last_chunk),
                    'labels': last_chunk.copy()
                }
            )
            last_chunk = []
            current_len = 0

    assert all(len(t['input_ids']) == MAX_SEQ_LENGTH for t in train_dataset_chunk[:-1]), f"Not all sequences are of length {MAX_SEQ_LENGTH} in chunk {dcix}"
    return train_dataset_chunk


dataset_chunks = [dataset.shard(num_shards=NUM_CHUNKS, index=i) for i in range(NUM_CHUNKS)]

with mp.Pool(processes=NUM_CHUNKS) as pool:
    process_chunk_with_index = partial(process_chunk)
    results = pool.starmap(process_chunk_with_index, [(chunk, i) for i, chunk in enumerate(dataset_chunks)])

train_dataset = []
for result in results:
    train_dataset.extend(result)

train_dataset = Dataset.from_list(train_dataset)
train_dataset = train_dataset.shuffle(seed=42)

hf_login(os.getenv("HF_TOKEN_EDWIN"))

train_dataset.push_to_hub(
    "edwindn/emilia-snac-orpheus-1b",
    split="train",
    private=False
)
