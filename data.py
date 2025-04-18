from datasets import load_dataset
import torch
from snac import SNAC
from transformers import AutoTokenizer
# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

REMOVE_DUPLICATES = False

# NOTE 
# ---------------------- #

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

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

snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac = snac.to(device)  # Move model to device

def encode_audio(audio):
    """
    must be a tensor of shape B, 1, T
    """
    with torch.inference_mode():
        codes = snac.encode(audio)

    c0 = codes[0].flatten()
    N = c0.size(0)

    c1 = codes[1].flatten().view(N, 2) + snac_vocab_size
    c2 = codes[2].flatten().view(N, 4) + snac_vocab_size * 2
    out = [
        c0,
        c1[:, 0],
        c2[:, 0],
        c2[:, 1],
        c1[:, 1],
        c2[:, 2],
        c2[:, 3]
    ]
    out = torch.stack(out, dim=1).flatten() # tensor of raw snac tokens

    # remove repeated tokens
    indices = torch.where(c0[:-1] == c0[1:])[0]
    if len(indices) > 0 and REMOVE_DUPLICATES:
        mask_indices = (indices.unsqueeze(1) * 7 + torch.arange(7, device=indices.device)).flatten()
        mask = torch.ones(len(out), dtype=torch.bool, device=out.device)
        mask[mask_indices] = False
        out = out[mask]

    out = out + audio_token_start # tensor of llama-ready tokens
    return out.to(device)  # Ensure output is on device


def dataset_map(batch):
    texts = batch["text"]
    audios = [item["array"] for item in batch["audio"]]  # list of np arrays
    audios = [torch.tensor(audio, dtype=torch.float32).reshape(1, 1, -1).to(device) for audio in audios]
    audio_tokens = [encode_audio(audio) for audio in audios]
    text_tokens = [tokenizer(text, return_tensors="pt").input_ids[0].to(device) for text in texts]  # Move text tokens to device
    
    start = torch.tensor([start_of_human, start_of_text], device=device)  # Create on device
    middle = torch.tensor([end_of_text, end_of_human, start_of_gpt, start_of_audio], device=device)  # Create on device
    end = torch.tensor([end_of_audio, end_of_gpt], device=device)  # Create on device

    tokens = [torch.cat([start, t, middle, a, end]) for a, t in zip(audio_tokens, text_tokens)]
    return {
            "tokens": tokens
        }

# ---------------------- #
# prepare data for DDP

MAX_DATA_LEN = 8192

def concat_tokens(batch):
    """
    for DDP training
    """
    raise NotImplementedError()
    all_tokens = []
    current_tokens = []
    current_length = 0
    
    for tokens in batch["tokens"]:
        token_length = len(tokens)
        
        if current_length + token_length > MAX_DATA_LEN:
            # Pad and add current sequence
            if current_tokens:
                padded = torch.cat(current_tokens)
                pad_length = MAX_DATA_LEN - len(padded)
                if pad_length > 0:
                    padded = torch.cat([padded, torch.full((pad_length,), pad_token, device=padded.device)])
                all_tokens.append(padded)
            
            # Start new sequence
            current_tokens = [tokens]
            current_length = token_length
        else:
            current_tokens.append(tokens)
            current_length += token_length
    
    # Handle remaining tokens
    if current_tokens:
        padded = torch.cat(current_tokens)
        pad_length = MAX_DATA_LEN - len(padded)
        if pad_length > 0:
            padded = torch.cat([padded, torch.full((pad_length,), pad_token, device=padded.device)])
        all_tokens.append(padded)
        
    return {"tokens": all_tokens}

if __name__ == "__main__":
    ds = load_dataset("canopylabs/zac-sample-dataset")['train']
    ds = ds.map(dataset_map, batched=True, remove_columns=ds.column_names)
    #ds = ds.map(concat_tokens, batched=True, remove_columns=ds.column_names)
    print(ds[0])

