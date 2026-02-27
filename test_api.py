import requests
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
url = "https://router.huggingface.co/hf-inference/models/openai-community/gpt2"
headers = {"Authorization": f"Bearer {token}"}

response = requests.post(url, headers=headers, json={"inputs": "The car is"})
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")