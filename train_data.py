from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
import torch
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download, login as hf_login
import os
import dotenv
from tqdm import tqdm

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
    #audio_tokens = torch.tensor(entry['codes_list'], dtype=torch.long)
    audio_tokens = entry['codes_list']
    text = entry['text']
    #text_tokens = tokenizer(text, return_tensors="pt").input_ids[0]
    text_tokens = tokenizer(text).input_ids

    #start = torch.tensor([start_of_human]) # tokenizer already adds sot token
    #middle = torch.tensor([end_of_text, end_of_human, start_of_gpt, start_of_audio])
    #end = torch.tensor([end_of_audio, end_of_gpt])
    #tokens = torch.cat([start, text_tokens, middle, audio_tokens, end])

    start = [start_of_human]
    middle = [end_of_text, end_of_human, start_of_gpt, start_of_audio]
    end = [end_of_audio, end_of_gpt]
    tokens = start + text_tokens + middle + audio_tokens + end
    return {"tokens": tokens}


dataset = dataset.map(tokenize_map, batched=False, remove_columns=dataset.column_names, num_proc=CPU_COUNT)

dataset = dataset.shuffle(seed=42)
dataset = dataset.batch(batch_size=1)
hf_login(os.getenv("HF_TOKEN_EDWIN"))

dataset.push_to_hub(
    "edwindn/emilia-snac-orpheus-1b-unpadded",
    split="train",
    private=True
)

quit()

# SEQ_LEN chunking
train_dataset = []

current_len = 0
last_chunk = []

for row in tqdm(dataset):
    tokens = row['tokens']
    
    while current_len + len(tokens) > MAX_SEQ_LENGTH:
        needed = MAX_SEQ_LENGTH - current_len
        last_chunk.extend(tokens[:needed])
        train_dataset.append(last_chunk)
        
        tokens = tokens[needed:]
        last_chunk = []
        current_len = 0
    
    last_chunk.extend(tokens)
    current_len += len(tokens)
    
    if current_len == MAX_SEQ_LENGTH:
        train_dataset.append(last_chunk)
        last_chunk = []
        current_len = 0

# Verify all sequences are the correct length
assert all(len(t) == MAX_SEQ_LENGTH for t in train_dataset[:-1]), f"Not all sequences are of length {MAX_SEQ_LENGTH}"

train_dataset = [{"tokens": t} for t in train_dataset]
train_dataset = Dataset.from_list(train_dataset)
train_dataset = train_dataset.shuffle(seed=42)
train_dataset = train_dataset.batch(batch_size=1)

hf_login(os.getenv("HF_TOKEN_EDWIN"))

train_dataset.push_to_hub(
    "edwindn/emilia-snac-orpheus-1b",
    split="train",
    private=True
)
