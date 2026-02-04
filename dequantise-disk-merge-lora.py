from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Coder-480B-A35B-Instruct")

# Load the LoRA adapter
model = PeftModel.from_pretrained(base_model, "/workspace/Qwen3-Coder-480B-A35B-Instruct-lora")

# Merge and unload
merged_model = model.merge_and_unload()

# Save the merged model
merged_model.save_pretrained("/workspace/Qwen3-Coder-480B-A35B-Instruct-heretic")
tokenizer.save_pretrained("/workspace/Qwen3-Coder-480B-A35B-Instruct-heretic")