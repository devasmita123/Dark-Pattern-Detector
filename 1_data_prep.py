# 2. Data Preparation Pipeline
# This script takes your existing e-commerce dataset, supplements it with synthetic legal/privacy patterns, and prepares it for training.

import pandas as pd
import random

def prepare_data():
    print("Loading Princeton dataset...")
    # Load the original dataset
    try:
        df_princeton = pd.read_csv('dark-patterns-v2.csv')
        df_princeton = df_princeton[['Pattern String', 'Deceptive?']].dropna()
        df_princeton.rename(columns={'Pattern String': 'text', 'Deceptive?': 'label'}, inplace=True)
        # Convert 'Yes'/'No' to 1 and 0
        df_princeton['label'] = df_princeton['label'].map({'Yes': 1, 'No': 0})
    except FileNotFoundError:
        print("Error: dark-patterns-v2.csv not found. Place it in the project root.")
        return

    print("Generating synthetic legal dataset...")
    # Synthetic legal dark patterns (Tricks)
    legal_tricks = [
        "By continuing to use this service, you waive your right to a class action lawsuit.",
        "We may share your data with our partners. Uncheck this hidden box in settings to opt out.",
        "You agree to let us record your keystrokes for quality purposes.",
        "Your subscription will automatically renew at $99/month unless cancelled 30 days in advance by phone.",
        "We reserve the right to change this privacy policy at any time without notifying you."
    ] * 20 # Multiplying to simulate a larger dataset

    # Synthetic fair legal clauses (Safe)
    legal_safe = [
        "We will never sell your personal information to third parties.",
        "You can delete your account and all associated data at any time from your profile page.",
        "We only collect your email address to send you password reset links.",
        "If we make material changes to this policy, we will notify you via email.",
        "Your payment information is encrypted and processed securely via Stripe."
    ] * 20

    # Create synthetic dataframe
    df_synthetic = pd.DataFrame({
        'text': legal_tricks + legal_safe,
        'label': [1]*len(legal_tricks) + [0]*len(legal_safe)
    })

    # Merge and shuffle
    df_final = pd.concat([df_princeton, df_synthetic]).sample(frac=1).reset_index(drop=True)
    
    # Save the new clean dataset
    df_final.to_csv('processed_training_data.csv', index=False)
    print(f"Data preparation complete! Total samples: {len(df_final)}")

if __name__ == "__main__":
    prepare_data()