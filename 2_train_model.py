# 3. Model Training (RoBERTa)
# This script trains your classifier to filter safe vs. tricky sentences. It saves the optimized model locally.

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding

# 1. Custom Dataset Class
class LegalDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long) # <--- ADD dtype=torch.long
        return item

    def __len__(self):
        return len(self.labels)

# 2. Evaluation Metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_roberta():
    print("Loading processed data...")
    df = pd.read_csv('processed_training_data.csv')
    
    # --- FIX START: Clean the data to remove NaNs ---
    # 1. Drop any rows where 'text' or 'label' is missing
    df = df.dropna(subset=['text', 'label'])
    
    # 2. Force text to be strings and labels to be integers
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    # --- FIX END ---

    # Split data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    print("Initializing RoBERTa Tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

    train_dataset = LegalDataset(train_encodings, train_labels)
    val_dataset = LegalDataset(val_encodings, val_labels)

    print("Initializing Model...")
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch", # Note: HuggingFace recently updated this from evaluation_strategy
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    print("Starting training...")
    trainer.train()

    print("Saving the final model...")
    model.save_pretrained('./legal_shield_model')
    tokenizer.save_pretrained('./legal_shield_model')
    print("Model saved to ./legal_shield_model")

if __name__ == "__main__":
    train_roberta()