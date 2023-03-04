from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
model_name = "distilgpt2"  # Use distilgpt2 or a fine-tuned version for mental health
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

# Define a function for generating responses
def generate_response(input_text, max_length=100, temperature=0.7, do_sample=True, top_p=0.9, top_k=50):
    # Encode input and generate response
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long)  # Create an attention mask

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            temperature=temperature,
            do_sample=do_sample,  # Enable sampling for more diverse responses
            top_p=top_p,
            top_k=top_k,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Chat loop
print("Welcome to the Mental Health Chatbot! Type 'quit' to end the chat.")
while True:
    user_input = input("You: ")
    
    if user_input.lower() == "quit":
        print("Chatbot: Take care! Remember, you're not alone.")
        break
    
    response = generate_response(user_input)
    print(f"Chatbot: {response}")
