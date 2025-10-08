"""
Test script to download and load google/gemma-3-12b-it
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Starting model download test...")
print("Model: google/gemma-3-12b-it")
print("=" * 80)

# Download tokenizer
print("\n[1/2] Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
print("✓ Tokenizer loaded successfully!")
print(f"  Vocab size: {tokenizer.vocab_size}")

# Download model
print("\n[2/2] Downloading model...")
print("  This may take 5-10 minutes for first download (~24GB)")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-3-12b-it",
    torch_dtype=torch.float16,
    device_map="cuda" if torch.cuda.is_available() else None
)
print("✓ Model loaded successfully!")
print(f"  Model type: {type(model)}")
print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

print("\n" + "=" * 80)
print("✓ Download test completed successfully!")
print("The model is now cached and will load faster next time.")
