# model.py
import torch
import torch.nn as nn
import transformers

# Simple 2-layer MLP
class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# GPT2 from Hugging Face 
class GPT2(nn.Module):
    def __init__(self, model_name="gpt2", device="cuda", precision="float32"):
        super().__init__()
        precision_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = precision_map.get(precision, torch.float32)
        
        kwargs = {
            "dtype": torch_dtype,
            "device_map": device if device == "cuda" else "cpu",
            "attn_implementation": "eager",
            "low_cpu_mem_usage": True,
        }
        
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name, **kwargs
        ).eval()
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids)
        return outputs.logits

# Quick test
if __name__ == "__main__":
    # Test TinyMLP
    model = TinyMLP()
    x = torch.randn(1, 10)
    print("TinyMLP output:", model(x).shape)
    
    # Test GPT2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpt2 = GPT2(model_name="gpt2", device=device, precision="float32")
    input_ids = torch.randint(0, 50257, (1, 32), device=device)
    print("GPT2 output:", gpt2(input_ids).shape)

