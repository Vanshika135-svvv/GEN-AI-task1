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

load_dotenv()
app = Flask(__name__)

# Initialize OpenAI client for Hugging Face Router
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
    if not HF_TOKEN:
        return jsonify({'text': "Auth Error: HF_TOKEN missing.", 'stats': 'Error'}), 500

    start_time = time.time()
    user_data = request.json
    user_prompt = user_data.get('prompt', '')
    
    # Logic to handle the "Long (Deep Generation)" dropdown from your UI
    generation_size = user_data.get('size', 'medium')
    
    # Map UI selection to token limits
    # Nanbeige 4.1-3B supports larger contexts for detailed studies
    token_limits = {
        "short": 150,
        "medium": 500,
        "long": 1500  # High limit for "Detailed Study" requests
    }
    max_tokens = token_limits.get(generation_size, 500)

    try:
        completion = client.chat.completions.create(
            model="Nanbeige/Nanbeige4.1-3B:featherless-ai",
            messages=[
                {
                    "role": "system", 
                    "content": "You are a highly detailed AI research assistant. Provide comprehensive, structured studies."
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=0.8,  # Higher creativity for long-form content
            top_p=0.95
        )

        raw_text = completion.choices[0].message.content
        
        # Clean up <think> tags if you don't want the internal reasoning in your UI
        # clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        clean_text = raw_text 

        duration = round(time.time() - start_time, 2)
        word_count = len(clean_text.split())

        return jsonify({
            'text': clean_text,
            'stats': f"{duration}s",
            'word_count': word_count
        })

    except Exception as e:
        print(f"ERROR: {str(e)}")
        return jsonify({'text': f"Generation Error: {str(e)}", 'stats': 'Fail'}), 200

if __name__ == '__main__':
    app.run()