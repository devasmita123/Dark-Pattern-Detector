import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from tqdm import tqdm

def evaluate_model():
    print("Loading data and model...")
    # 1. Load and clean data exactly as we did in training
    df = pd.read_csv('processed_training_data.csv')
    df = df.dropna(subset=['text', 'label'])
    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)

    # 2. Use the exact same random_state (42) so we get the exact same testing data
    _, val_texts, _, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42
    )

    # 3. Load the trained model
    model_path = './legal_shield_model'
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    model.eval() # Put model in evaluation mode

    print(f"Running predictions on {len(val_texts)} test sentences...")
    predictions = []

    # 4. Run inference
    for text in tqdm(val_texts):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=-1).item()
        predictions.append(pred)

    # 5. Print the official metrics
    print("\n" + "="*50)
    print("FINAL MODEL METRICS")
    print("="*50)
    print(classification_report(val_labels, predictions, target_names=["Safe (0)", "Dark Pattern (1)"]))

    # 6. Generate and save the Confusion Matrix plot
    print("\nGenerating Confusion Matrix graph...")
    cm = confusion_matrix(val_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=["Safe", "Dark Pattern"], 
                yticklabels=["Safe", "Dark Pattern"])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Model Performance: Confusion Matrix')
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ Saved 'confusion_matrix.png' to your project folder!")

if __name__ == "__main__":
    evaluate_model()