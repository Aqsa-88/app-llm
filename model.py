from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"  

# Downloads and stores model in ./model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
# Generate text
input_text = "Once upon a time,"
input = tokenizer(input_text, return_tensors = "pt")

output = model.generate(**input,max_lenght= 50)

print(tokenizer.decode(output[0],skip_special_tokens = True))

