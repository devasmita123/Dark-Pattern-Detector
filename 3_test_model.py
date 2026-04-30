import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def load_model():
    print("Loading your trained Legal Shield model...")
    # This points to the folder where your model was saved in the previous step
    model_path = './legal_shield_model' 
    
    tokenizer = RobertaTokenizer.from_pretrained(model_path)
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model

def predict(text, tokenizer, model):
    # 1. Prepare the text the same way we did for training
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    # 2. Pass the text through the model (no_grad because we are just testing, not training)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # 3. Calculate probabilities using softmax
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    # 4. Get the highest probability class
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    confidence = probabilities[0][predicted_class].item()
    
    # Based on your dataset: 1.0 is usually a Dark Pattern, 0.0 is Safe
    label = "🚨 Dark Pattern / Tricky" if predicted_class == 1 else "✅ Safe / Normal"
    
    return label, confidence

if __name__ == "__main__":
    tokenizer, model = load_model()
    
    print("\n" + "="*50)
    print("🤖 MODEL READY! Type a sentence to test it.")
    print("Type 'quit' to exit.")
    print("="*50)
    
    while True:
        user_input = input("\nEnter a sentence: ")
        
        if user_input.lower() == 'quit':
            print("Shutting down. Goodbye!")
            break
            
        if user_input.strip() == "":
            continue
            
        label, confidence = predict(user_input, tokenizer, model)
        print(f"Result: {label}")
        print(f"Confidence: {confidence:.2%}")