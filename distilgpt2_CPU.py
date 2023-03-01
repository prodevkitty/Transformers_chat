from transformers import pipeline

print("Loading the DistilGPT-2 model...")
chatbot = pipeline("text-generation", model="distilgpt2")

print("Welcome to the Mental Health Chatbot! Type 'quit' to end the chat.")

while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        print("Chatbot: Take care! Have a great day!")
        break

    response = chatbot(user_input, max_length=100, num_return_sequences=1, do_sample=True)
    print("Chatbot:", response[0]["generated_text"])
