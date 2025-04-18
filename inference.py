import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn


tokenizer = AutoTokenizer.from_pretrained("edwindn/emilia-snac-orpheus-1b")
model = AutoModelForCausalLM.from_pretrained("edwindn/emilia-snac-orpheus-1b")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)

class OrpheusInference(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)


