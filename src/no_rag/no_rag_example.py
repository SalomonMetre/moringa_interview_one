from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

#query
QUERY = "Who is Harriet ?"

# Prompt using chat template
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": QUERY}],
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# Generate response
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True)

# Decode and print
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)  # clean output
