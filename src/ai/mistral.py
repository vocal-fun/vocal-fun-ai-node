from transformers import AutoModelForCausalLM, AutoTokenizer

# # Load the tokenizer and model
# model_name = "mistralai/Mistral-7B-v0.1"  # Replace with the specific Mistral model
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="float16")

# # Function to generate responses
# def generate_response_mistral(prompt, max_new_tokens=100):
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Send inputs to GPU
#     outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

