from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Load the trained model and tokenizer
model_path = "./chatbot_model_two_t5_base_continued/checkpoint-45000"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Function to chat with the model
def chat():
    print("Chatbot is ready! Type 'quit' to exit.")
    print("Bot: Hello! How can I help you?")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        # Use prompt format to reduce repetition
        prompt = f"User: {user_input}\nBot:"

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        output_tokens = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=5,
            early_stopping=True
        )

        bot_response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        # Optional: remove repeated user input
        def remove_repeated_user_input(user_input, bot_response):
            user_input = user_input.strip().lower()
            response = bot_response.strip()
            if response.lower().startswith(user_input):
                response = response[len(user_input):].lstrip(" ,.:!?")
            return response

        cleaned_response = remove_repeated_user_input(user_input, bot_response)
        print(f"Bot: {cleaned_response}")


# Start chatting
chat()