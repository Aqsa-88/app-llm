from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "gpt2"  


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

app = Flask(__name__)

@app.route("/generate", methods=["POST"])
def generate():
    data = request.json  # Receive JSON from Postman
    prompt = data.get("text", "")
    
    if not prompt:
        return jsonify({"error": "Text is required"}), 400

    # Just return the prompt with "Generated: " for now
    return jsonify({"generated_text": "Generated: " + prompt})
    # try:
        # data = request.json
        # prompt = data.json.get("text", "")
        # max_lenght = data.get("max_lenght",50)

        # if not prompt:
        #     return jsonify({"error": "Text is required"}), 400

        # inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        # outputs = model.generate(**inputs, max_length=max_lenght)
        # text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    #     return jsonify({"generated_text": text})
    
    # except Exception as e:
    #     return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)






