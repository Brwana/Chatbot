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

# Load the dataset
dataset = load_dataset("multi_woz_v22")

# Filter for only "hotel" service
# Filter for only hotel, restaurant, and taxi services
train_data = [dialog for dialog in dataset['train'] if
              any(service in dialog['services'] for service in ["hotel", "restaurant", "taxi"])]
test_data = [dialog for dialog in dataset['test'] if
             any(service in dialog['services'] for service in ["hotel", "restaurant", "taxi"])]

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# --- 🔵 IMPROVED: preprocess_text (keep stopwords) ---
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    return " ".join(tokens)

# Load the model and tokenizer
local_model_path = r"C:\Users\nourn\my_models\t5-base"  # Change this to your local model path
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)

# --- 🔵 IMPROVED: prepare_conversation_data (only preprocess users + use more context) ---
def prepare_conversation_data(dialog):
    """Convert a dialogue into input-output pairs for training"""
    conversation = []
    current_context = []

    for i in range(len(dialog['turns']['utterance'])):
        utterance = dialog['turns']['utterance'][i]
        speaker = dialog['turns']['speaker'][i]

        if speaker == 0:  # User
            processed_utterance = preprocess_text(utterance)
            current_context.append(processed_utterance)
        else:  # System
            processed_utterance = utterance  # Keep system response raw (no preprocessing)
            if current_context:
                context = " [SEP] ".join(current_context[-6:])  # Use more past turns
                conversation.append({
                    'context': context,
                    'response': processed_utterance,
                    'dialogue_acts': dialog['turns']['dialogue_acts'][i] if 'dialogue_acts' in dialog['turns'] else []
                })

    return conversation

# Prepare training and testing conversations
train_conversations = []
for dialog in train_data:
    train_conversations.extend(prepare_conversation_data(dialog))

test_conversations = []
for dialog in test_data:
    test_conversations.extend(prepare_conversation_data(dialog))

# Convert to DataFrame
train_df = pd.DataFrame(train_conversations)
test_df = pd.DataFrame(test_conversations)

# --- 🔵 IMPROVED: use max_length=256 ---
class ConversationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=256):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context = str(self.data.iloc[index]['context'])
        response = str(self.data.iloc[index]['response'])

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

# --- 🔵 IMPROVED: num_train_epochs = 6 ---
training_args = Seq2SeqTrainingArguments(
    output_dir='./chatbot_model_t5_base',
    num_train_epochs=6,
    per_device_train_batch_size=4,  # ↓ to fit into memory
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=8,  # higher to simulate batch size of 32
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=1000,
    save_total_limit=2,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none"
)


# Create trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,

)

# Start training
trainer.train()

# Save model
trainer.save_model("./chatbot_model_two/final_model")
tokenizer.save_pretrained("./chatbot_model_two/final_model")
