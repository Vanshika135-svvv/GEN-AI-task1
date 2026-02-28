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
import time
from flask import Flask, render_template, request, jsonify
from openai import OpenAI
from dotenv import load_dotenv

# Load local .env file
load_dotenv()

app = Flask(__name__)

# --- 2026 ROUTER CONFIGURATION ---
# Initialize the OpenAI-compatible client for Hugging Face
HF_TOKEN = os.getenv("HF_TOKEN")
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    # Safety Check: Ensure token is present
    if not HF_TOKEN:
        return jsonify({'text': "Error: HF_TOKEN missing in Vercel Settings.", 'stats': 'Config Error'}), 500

    start_time = time.time()
    user_data = request.json
    user_prompt = user_data.get('prompt', '')

    try:
        # Use the specific model and syntax you provided
        completion = client.chat.completions.create(
            model="Nanbeige/Nanbeige4.1-3B:featherless-ai",
            messages=[
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=150 # Limits response length for the UI
        )

        # Extract the generated message content
        raw_text = completion.choices[0].message.content
        
        duration = round(time.time() - start_time, 2)
        word_count = len(raw_text.split())

        return jsonify({
            'text': raw_text,
            'stats': f"{duration}s",
            'word_count': word_count
        })

    except Exception as e:
        # Catch errors like "Invalid API Key" or "Model Not Found"
        print(f"ERROR: {str(e)}")
        return jsonify({'text': f"API Error: {str(e)}", 'stats': 'Fail'}), 200

if __name__ == '__main__':
    app.run()