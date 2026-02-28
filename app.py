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
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)

# --- 2026 PRODUCTION CONFIGURATION ---
# Switching to SmolLM2 because legacy GPT-2 often returns 404 on the new router
API_URL = "https://router.huggingface.co/hf-inference/models/HuggingFaceTB/SmolLM2-135M-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # DEBUG: Logs token status to Vercel/Local terminal
    if HF_TOKEN:
        print(f"DEBUG: Token Active ({HF_TOKEN[:5]}...)") 
    else:
        print("DEBUG: TOKEN IS MISSING!")

    if not HF_TOKEN:
        return jsonify({'text': "Auth Error: HF_TOKEN missing in settings.", 'stats': 'Error'}), 500

    start_time = time.time()
    data = request.json
    prompt = data.get('prompt', '')
    
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 100, "do_sample": True, "temperature": 0.7}
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        
        # If API returns 404, it means this model isn't on the router's active list
        if response.status_code != 200:
            return jsonify({'text': f"API Error {response.status_code}: {response.text}", 'stats': 'Fail'}), 200
            
        result = response.json()
        
        # Extract text safely
        if isinstance(result, list) and len(result) > 0:
            raw_text = result[0].get('generated_text', 'No text generated.')
        else:
            raw_text = result.get('generated_text', 'No text generated.')

        return jsonify({
            'text': raw_text,
            'stats': f"{round(time.time() - start_time, 2)}s",
            'word_count': len(raw_text.split())
        })
    except Exception as e:
        return jsonify({'text': f"System Error: {str(e)}", 'stats': 'Crash'}), 500

if __name__ == '__main__':
    app.run()