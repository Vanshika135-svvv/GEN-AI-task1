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
import os
import requests
import time
import re
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

# Load local .env file
load_dotenv()

app = Flask(__name__)

# OPTION A: DistilGPT2 (Lightweight version of GPT-2)
API_URL = "https://router.huggingface.co/hf-inference/models/Qwen/Qwen3.5-35B-A3B"

# OPTION B: Gemma (More modern and reliable)
# API_URL = "https://router.huggingface.co/hf-inference/models/google/gemma-2-2b-it"

HF_TOKEN = os.getenv("HF_TOKEN")
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    start_time = time.time()
    data = request.json
    prompt = data.get('prompt', '')
    size = data.get('size', 'medium')

    # API specific parameters for depth control
    config = {
        "short": {"max_new_tokens": 50, "repetition_penalty": 1.1},
        "medium": {"max_new_tokens": 150, "repetition_penalty": 1.2},
        "long": {"max_new_tokens": 250, "repetition_penalty": 1.3} 
    }
    
    sel = config.get(size, config["medium"])

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": sel["max_new_tokens"],
            "repetition_penalty": sel["repetition_penalty"],
            "do_sample": True,
            "temperature": 0.8,
            "top_p": 0.9,
            "return_full_text": False 
        }
    }

    try:
        # Call the Hugging Face Router API
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        
        # Extract generated text
        raw_text = result[0]['generated_text'] if isinstance(result, list) else "Error."
        
        # --- SMART SENTENCE CLEANER ---
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

    except Exception as e:
        # Returns specific error message to help debug
        return jsonify({'text': f"API Error: {str(e)}", 'stats': 'Error'}), 500

if __name__ == '__main__':
    app.run()