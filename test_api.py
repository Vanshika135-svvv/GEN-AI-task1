import requests
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
# Try the updated URL here
url = "https://router.huggingface.co/hf-inference/models/google/gemma-2-2b-it"
headers = {"Authorization": f"Bearer {token}"}

response = requests.post(url, headers=headers, json={"inputs": "The future of AI is"})

print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    print(f"Success! Response: {response.json()}")
else:
    print(f"Failed with error: {response.text}")