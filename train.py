import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from huggingface_hub import login as hf_login, snapshot_download
import wandb
import dotenv

dotenv.load_dotenv()

"""
run on gpu

accelerate launch train.py

move the map function to train_data.py
"""

hf_login(os.getenv("HF_TOKEN_EDWIN"))

USE_WANDB = False

if USE_WANDB:
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

# ---------------------- #

# Download dataset
dataset_path = snapshot_download(
    repo_id="edwindn/emilia-snac-orpheus-1b-test",
    repo_type="dataset",
    revision="main",
    max_workers=CPU_COUNT,
)

dataset = load_dataset(dataset_path, split="train")

# SAMPLE FOR TESTING
dataset = dataset.select(range(100))

print([len(x['tokens']) for x in dataset])
print(dataset[0])

# Format dataset for model input
def format_dataset(example):
    return {
        'input_ids': example['tokens'],
        'attention_mask': [1] * len(example['tokens']),
        'labels': example['tokens']
    }

dataset = dataset.map(format_dataset)

# Setup training arguments with DDP
training_args = TrainingArguments(
    output_dir="checkpoints",
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    num_train_epochs=1,
    learning_rate=2e-5,
    weight_decay=0.01,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    save_total_limit=3,
    eval_steps=100,
    ddp_find_unused_parameters=False,
    ddp_timeout=1800,
    local_rank=int(os.environ.get("LOCAL_RANK", -1)),
    report_to="wandb" if USE_WANDB else None
)

# Create data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train
trainer.train()

# Push model to Hugging Face Hub
trainer.push_to_hub("edwindn/orpheus-1b-0.1", private=True)

