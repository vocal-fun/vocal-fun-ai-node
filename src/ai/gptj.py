from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "EleutherAI/gpt-j-6B"

# # Load model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype="float16")

# # Generate response
# def generate_response_gptj(prompt, max_new_tokens=100):
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)