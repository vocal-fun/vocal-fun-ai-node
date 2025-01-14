from transformers import AutoModelForCausalLM, AutoTokenizer

# Set model name
model_name = "bigscience/bloom-560m"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto"  # Automatically handles device placement
)

# Function to generate responses
def generate_response_bloom(prompt, max_new_tokens=50):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)  # Ensure inputs are sent to the correct device
    outputs = model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
prompt = "What is the capital of France?"
response = generate_response_bloom(prompt)
print("AI:", response)
