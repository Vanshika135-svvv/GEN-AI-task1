# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# # 1. Setup Device (Uses GPU if available, else CPU)
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Running on: {device}")

# # 2. Load Model and Tokenizer
# # This will download about 500MB the first time you run it
# print("Loading model...")
# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# # 3. Your Prompt
# prompt = "The future of Artificial Intelligence is"

# # 4. Convert text to numbers (Tensors)
# inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

# # 5. Generate Text (Using 'Top-P' sampling for better results)
# print("Generating...")
# output = model.generate(
#     inputs, 
#     max_length=100,            # Length of the output
#     do_sample=True,            # Enable creative sampling
#     top_p=0.95,                # Nucleus sampling
#     top_k=50,                  # Top-K sampling
#     temperature=0.8,           # Creativity level
#     no_repeat_ngram_size=2     # Prevents loops
# )

# # 6. Convert numbers back to text
# result = tokenizer.decode(output[0], skip_special_tokens=True)

# print("\n--- RESULT ---")
# print(result)
from flask import Flask, render_template, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import time
import re
import torch

app = Flask(__name__)

# Load model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    start_time = time.time()
    data = request.json
    prompt = data.get('prompt', '')
    size = data.get('size', 'medium')

    # Configuration for true length depth
    # max_length: absolute ceiling | min_length: forced generation
    config = {
        "short": {"max": 80, "min": 30},
        "medium": {"max": 250, "min": 100},
        "long": {"max": 500, "min": 250} 
    }
    
    sel = config.get(size, config["medium"])
    inputs = tokenizer.encode(prompt, return_tensors="pt")

    # Advanced generation for long-form content
    outputs = model.generate(
        inputs, 
        max_length=sel["max"],
        min_length=sel["min"],
        do_sample=True, 
        top_p=0.95, 
        temperature=0.85, 
        no_repeat_ngram_size=3,    # Higher value allows more natural flow
        repetition_penalty=1.2,    # Prevents "looping" on long text
        pad_token_id=tokenizer.eos_token_id
    )
    
    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # --- SMART SENTENCE CLEANER ---
    # We only trim if the text ends abruptly without punctuation
    if not raw_text.endswith(('.', '!', '?')):
        match = list(re.finditer(r'[.!?]', raw_text))
        if match:
            last_index = match[-1].start()
            clean_text = raw_text[:last_index + 1]
        else:
            clean_text = raw_text
    else:
        clean_text = raw_text

    duration = round(time.time() - start_time, 2)
    word_count = len(clean_text.split())

    return jsonify({
        'text': clean_text,
        'stats': f"{duration}s",
        'word_count': word_count
    })

if __name__ == '__main__':
    app.run(debug=True)