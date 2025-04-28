# Import necessary libraries
from datasets import load_dataset
import nltk
import numpy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os

# Load the dataset from the local path
dataset = load_dataset("multi_woz_v22")


# Filter for only "hotel" service in both training and testing splits
train_data = [dialog for dialog in dataset['train'] if "hotel" in dialog['services']]
test_data = [dialog for dialog in dataset['test'] if "hotel" in dialog['services']]

# Function to print relevant user and bot turns along with slots
def print_hotel_dialogs(data_split):
    for dialog in data_split:
        print(f"Dialogue ID: {dialog['dialogue_id']}")

        # Debug: Print the structure of the entire dialogue to understand its format
        print(f"Dialogue Structure: {dialog}")

        # Check if 'turns' exists and has data
        if 'turns' in dialog and len(dialog['turns']['utterance']) > 0:
            print(f"First turn in this dialogue: {dialog['turns']['utterance'][0]}")

            # Iterate through turns
            num_turns = len(dialog['turns']['utterance'])
            for i in range(num_turns):
                # Extract details of each turn
                speaker = dialog['turns']['speaker'][i]
                utterance = dialog['turns']['utterance'][i]
                print(f"Turn {i + 1} - Speaker: {speaker}, Utterance: {utterance}")

                # Extract relevant dialogue acts (e.g., Hotel-Inform, Hotel-Request)
                if 'dialogue_acts' in dialog and len(dialog['dialogue_acts']) > i:
                    acts = dialog['dialogue_acts'][i]
                    if 'dialog_act' in acts:
                        for act in acts['dialog_act']:
                            print(f"Act Type: {act['act_type']}")
                            print(f"Act Slots: {act['act_slots']}")

                    # If it's a user turn, print the user's belief state (slots)
                    if speaker == 0 and 'frames' in dialog and len(dialog['frames']) > i:
                        user_frame = dialog['frames'][i]
                        if 'slots' in user_frame:
                            print(f"Requested Slots: {user_frame['slots']}")
                else:
                    print(f"No dialogue acts available for turn {i + 1}")
        else:
            print(f"No turns available for dialogue {dialog['dialogue_id']}")

        print("\n")


# Example: Print the first 2 filtered hotel dialogues from the training set
print("Training Data:")
print_hotel_dialogs(train_data[:2])

# Example: Print the first 2 filtered hotel dialogues from the testing set
print("Testing Data:")
print_hotel_dialogs(test_data[:2])

# Download the necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    # Tokenize the input text
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase

    # Remove stopwords and non-alphabetic tokens, and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]

    # Return preprocessed tokens
    return " ".join(tokens)


# Example usage:
utterance = "i need a place to dine in the center thats expensive"
preprocessed_utterance = preprocess_text(utterance)
print(preprocessed_utterance)

# Load the model and tokenizer from local path
local_model_path = r"C:\Users\nourn\my_models\t5-small"  # Change this to your local model path
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)

def prepare_conversation_data(dialog):
    """Convert a dialogue into input-output pairs for training"""
    conversation = []
    current_context = []

    for i in range(len(dialog['turns']['utterance'])):
        utterance = dialog['turns']['utterance'][i]
        speaker = dialog['turns']['speaker'][i]

        # Preprocess the utterance
        processed_utterance = preprocess_text(utterance)

        if speaker == 0:  # User
            current_context.append(processed_utterance)
        else:  # System
            if current_context:  # Only if there's previous context
                # Combine previous turns as context
                context = " [SEP] ".join(current_context[-3:])  # Use last 3 turns as context
                conversation.append({
                    'context': context,
                    'response': processed_utterance,
                    'dialogue_acts': dialog['turns']['dialogue_acts'][i] if 'dialogue_acts' in dialog['turns'] else []
                })

    return conversation


# Prepare training data
train_conversations = []
for dialog in train_data:
    train_conversations.extend(prepare_conversation_data(dialog))

# Prepare test data
test_conversations = []
for dialog in test_data:
    test_conversations.extend(prepare_conversation_data(dialog))

# Convert to DataFrame for easier handling
train_df = pd.DataFrame(train_conversations)
test_df = pd.DataFrame(test_conversations)


# Create custom dataset for conversation data
class ConversationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context = str(self.data.iloc[index]['context'])
        response = str(self.data.iloc[index]['response'])

        # Tokenize context and response
        context_encoding = self.tokenizer(
            context,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        response_encoding = self.tokenizer(
            response,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': context_encoding['input_ids'].flatten(),
            'attention_mask': context_encoding['attention_mask'].flatten(),
            'labels': response_encoding['input_ids'].flatten()
        }


# Create datasets
train_dataset = ConversationDataset(train_df, tokenizer)
test_dataset = ConversationDataset(test_df, tokenizer)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./chatbot_model',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4,  # simulates batch of 32
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="steps",  # Correct parameter name
    eval_steps=500,
    save_steps=1000,
    report_to="none",  # Disables all logging integrations
    predict_with_generate=True,
    fp16=torch.cuda.is_available()
)

# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()
# Save the final model
trainer.save_model("./chatbot_model/final_model")  # Save model
tokenizer.save_pretrained("./chatbot_model/final_model")  # Save tokenizer

