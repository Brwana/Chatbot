# Import necessary libraries
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from torch.utils.data import Dataset
import torch
import numpy as np
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
import os
from nltk.translate import bleu_score  # Correct import for BLEU score
from evaluate import load  # Correct import for ROUGE score
from transformers import EarlyStoppingCallback

# Load ROUGE metric
rouge = load("rouge")
nltk.download('punkt_tab')

# Load the dataset
dataset = load_dataset("multi_woz_v22")

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

# --- 🔵 Preprocess user utterances ---
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha()]
    return " ".join(tokens)

# Load the model and tokenizer
# local_model_path = r"C:\Users\nourn\my_models\t5-base"  # Use t5-base or t5-large
# tokenizer = AutoTokenizer.from_pretrained(local_model_path)
# model = AutoModelForSeq2SeqLM.from_pretrained(local_model_path)
model = AutoModelForSeq2SeqLM.from_pretrained("./chatbot_model_two_t5_base/final_model")
tokenizer = AutoTokenizer.from_pretrained("./chatbot_model_two_t5_base/final_model")
# --- 🔵 Prepare dialog context-response pairs ---
def prepare_conversation_data(dialog):
    conversation = []
    current_context = []

    for i in range(len(dialog['turns']['utterance'])):
        utterance = dialog['turns']['utterance'][i]
        speaker = dialog['turns']['speaker'][i]

        if speaker == 0:  # User
            processed_utterance = preprocess_text(utterance)
            current_context.append("Language: English | " + processed_utterance)
        else:  # System
            processed_utterance = utterance  # Keep system response raw
            if current_context:
                context = " [SEP] ".join(current_context[-6:])
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

# --- 🔵 Dataset class with correct label masking ---
class ConversationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_length=128, max_output_length=64):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        context = str(self.data.iloc[index]['context'])
        response = str(self.data.iloc[index]['response'])

        context_encoding = self.tokenizer(
            context,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        response_encoding = self.tokenizer(
            response,
            max_length=self.max_output_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = response_encoding['input_ids'].flatten()
        labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding in loss

        return {
            'input_ids': context_encoding['input_ids'].flatten(),
            'attention_mask': context_encoding['attention_mask'].flatten(),
            'labels': labels
        }

# Create datasets
train_dataset = ConversationDataset(train_df, tokenizer)
test_dataset = ConversationDataset(test_df, tokenizer)
training_args = Seq2SeqTrainingArguments(
    output_dir="./chatbot_model_two_t5_base_continued",
    num_train_epochs=10,  # set to total desired (e.g. 10 if you already did 6)
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    eval_strategy="steps",
    eval_steps=500,
    logging_dir="./logs",
    logging_steps=500,
    save_steps=5000,
    report_to="tensorboard",
    learning_rate=1e-5,
    predict_with_generate=True,
    load_best_model_at_end=True,
)



# --- 🔵 Compute ROUGE scores ---
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def compute_metrics(pred):
    preds = np.array(pred.predictions)
    preds = np.where(preds < 0, tokenizer.pad_token_id, preds)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(pred.label_ids == -100, tokenizer.pad_token_id, pred.label_ids)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    smoothie = SmoothingFunction().method4
    scores = [
        sentence_bleu([label.split()], pred.split(), smoothing_function=smoothie)
        for pred, label in zip(decoded_preds, decoded_labels)
    ]
    avg_bleu = np.mean(scores)

    return {"bleu": avg_bleu}

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,  # reuse your existing dataset
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save model and tokenizer
trainer.save_model("./chatbot_model_two_t5_base_continued/final_model")
tokenizer.save_pretrained("./chatbot_model_two_t5_base_continued/final_model")