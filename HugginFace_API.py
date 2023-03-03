import time
import requests

API_TOKEN = "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"
headers = {"Authorization": f"Bearer {API_TOKEN}"}
url = "https://api-inference.huggingface.co/models/fadodr/finetuned_mental_health_therapy_original"

while True:
    response = requests.post(url, headers=headers, json={"inputs": "How are you feeling today?"})
    if response.status_code == 200:
        print(response.json())  # Successfully loaded and responded
        break
    elif response.status_code == 503:
        # Model is still loading
        print("Model is still loading. Retrying in 30 seconds...")
        time.sleep(30)
    else:
        print("Error:", response.json())
        break