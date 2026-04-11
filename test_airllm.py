import torch
from airllm import AirLLMLlama2

# Model ID for Qwen2.5-Coder-7B-Instruct
# Qwen2.5 uses the same architecture as Llama (with some minor variations compatible with AirLLMLlama2)
model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"

print(f"Loading model: {model_id} via AirLLM (layer-wise)...")

# Initialize AirLLM
# This will download the model if not present and load layers one by one during inference
model = AirLLMLlama2(model_id)

# Input prompt
input_text = "Task: Write a Python function to check if a number is prime.\n\nPython Code:"

print("Tokenizing and generating...")

# Tokenize
input_tokens = model.tokenizer(input_text, return_tensors="pt", return_attention_mask=False).input_ids.cuda()

# Generate
# We use moderate max_new_tokens for a quick verification
generation_output = model.generate(input_tokens, max_new_tokens=200, use_cache=True)

# Decode and print
output_text = model.tokenizer.decode(generation_output[0], skip_special_tokens=True)
print("\n--- GENERATED CODE ---\n")
print(output_text)
print("\n--- END ---")
