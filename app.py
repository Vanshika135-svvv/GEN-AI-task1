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

# Load environment variables (safely ignores missing .env on Vercel)
load_dotenv()

app = Flask(__name__)

# --- CONFIGURATION ---
# We use the verified full namespace for GPT-2
API_URL = "https://router.huggingface.co/hf-inference/models/openai-community/gpt2"

# Retrieve token from Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # --- SAFETY CHECK 1: CHECK TOKEN ---
    if not HF_TOKEN:
        return jsonify({
            'text': "Configuration Error: HF_TOKEN is missing. Please add it to Vercel Settings.",
            'stats': 'Config Error'
        }), 500

    start_time = time.time()
    data = request.json
    prompt = data.get('prompt', '')
    size = data.get('size', 'medium')

    # Define generation parameters
    config = {
        "short": {"max_new_tokens": 50},
        "medium": {"max_new_tokens": 150},
        "long": {"max_new_tokens": 250} 
    }
    sel = config.get(size, config["medium"])

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": sel["max_new_tokens"],
            "return_full_text": False,
            "do_sample": True,
            "temperature": 0.8
        }
    }

    try:
        # --- API CALL ---
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # --- SAFETY CHECK 2: HANDLE API ERRORS ---
        if response.status_code != 200:
            error_msg = f"API Error {response.status_code}: {response.text}"
            print(error_msg) # Logs to Vercel Console
            return jsonify({'text': error_msg, 'stats': 'API Fail'}), response.status_code
            
        result = response.json()
        
        # Handle list or dict response format
        if isinstance(result, list) and len(result) > 0:
            raw_text = result[0].get('generated_text', '')
        elif isinstance(result, dict):
            raw_text = result.get('generated_text', '')
        else:
            raw_text = "No text returned."

        # --- SMART CLEANER ---
        # Ensures text ends with punctuation
        if raw_text and not raw_text.endswith(('.', '!', '?')):
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
        # Catch unexpected crashes and return them as JSON
        print(f"CRASH: {str(e)}")
        return jsonify({'text': f"Server Crash: {str(e)}", 'stats': 'Crash'}), 500

if __name__ == '__main__':
    app.run()