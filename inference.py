import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from typing import Optional
from snac import SNAC
import numpy as np
from scipy.io.wavfile import write

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained("edwindn/emilia-snac-orpheus-1b")
model = model.to(device)

snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac = snac.to(device)

snac_sample_rate = 24e3

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


class OrpheusInference(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")


    def forward(
            self,
            text: str,
    ):
        
        input_ids = self.tokenizer(text, return_tensors="pt").input_ids[0]
        input_ids = [start_of_human, start_of_text] + input_ids + [end_of_text, end_of_human, start_of_gpt, start_of_audio]
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(self.device)

        out_tokens = self.model(input_ids).flatten()
        assert out_tokens[-1] == end_of_audio
        out_tokens = out_tokens[:-1]
        assert len(out_tokens) % 7 == 0, "Token length must be divisible by 7"
        
        out_tokens = out_tokens - audio_token_start
        out_tokens = out_tokens.reshape(-1, 7)
        
        snac0 = out_tokens[:, 0]
        snac1 = torch.cat([out_tokens[:, 1] - snac_vocab_size, out_tokens[:, 4] - snac_vocab_size * 4], dim=1)
        snac2 = torch.cat([out_tokens[:, 2] - snac_vocab_size * 2, out_tokens[:, 3] - snac_vocab_size * 3, out_tokens[:, 5] - snac_vocab_size * 5, out_tokens[:, 6] - snac_vocab_size * 6], dim=1)
        
        codes = [snac0.tolist(), snac1.tolist(), snac2.tolist()]
        assert all(c < snac_vocab_size for c in codes[0]), "snac0 must be less than snac_vocab_size"
        assert all(c < snac_vocab_size for c in codes[1]), "snac1 must be less than snac_vocab_size"
        assert all(c < snac_vocab_size for c in codes[2]), "snac2 must be less than snac_vocab_size"

        with torch.inference_mode():
            reconstructed_audio = snac.decode(codes)

        return reconstructed_audio
        


if __name__ == "__main__":
    orpheus = OrpheusInference(model, device)
    sample_text = "Hello, how are you?"
    reconstructed_audio = orpheus(sample_text)

    write("reconstructed_audio.wav", snac_sample_rate, reconstructed_audio)
    