import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from huggingface_hub import login as hf_login, snapshot_download
import wandb

"""
run on gpu

accelerate launch train.py
"""

hf_login(os.getenv("HF_TOKEN_EDWIN"))

# Initialize wandb
wandb.init(
    project="orpheus-1b",
    name="training-run",
    config={
        "model_name": "meta-llama/Llama-3.2-1B",
        "max_seq_length": 8192,
        "batch_size": 1,
        "learning_rate": 2e-5,
        "epochs": 1
    }
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

MAX_SEQ_LENGTH = 8192
CPU_COUNT = os.cpu_count()
TRAIN_BATCH_SIZE = 1

# ---------------------- #

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
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

# ---------------------- #

# Download dataset
dataset_path = snapshot_download(
    repo_id="edwindn/emilia-snac-orpheus-1b-unpadded",
    repo_type="dataset",
    revision="main",
    max_workers=CPU_COUNT,
)

dataset = load_dataset(dataset_path, split="train")

# Setup training arguments with DDP
training_args = TrainingArguments(
    output_dir="checkpoints",  # Directory for checkpoints
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    logging_steps=10,
    save_steps=100,  # Save a checkpoint every 100 steps
    save_total_limit=3,  # Keep only the last 3 checkpoints
    evaluation_strategy="steps",
    eval_steps=100,
    # DDP specific settings
    ddp_find_unused_parameters=False,
    ddp_timeout=1800,  # 30 minutes
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    report_to="wandb"  # Add wandb reporting
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="tokens",
    dataset_num_proc=4,
    packing=True,
    max_seq_length=MAX_SEQ_LENGTH,
    args=training_args,
)

# Train
trainer.train()

# Push model to Hugging Face Hub
trainer.push_to_hub("edwindn/orpheus-1b-0.1", private=True)

