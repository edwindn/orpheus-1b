import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from typing import Optional
from snac import SNAC


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = AutoModelForCausalLM.from_pretrained("edwindn/emilia-snac-orpheus-1b")
model = model.to(device)

snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
snac = snac.to(device)

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

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None
    ):
        out_tokens = self.model(input_ids, attention_mask).flatten()
        out_tokens = out_tokens - llama_token_end
        # extract between start_of_audio and end_of_audio tokens

        
        
        

        snac_row0 = out



if __name__ == "__main__":
    orpheus = OrpheusInference(model, device)